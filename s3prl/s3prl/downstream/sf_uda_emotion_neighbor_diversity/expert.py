import os
import math
import torch
import random
from pathlib import Path
from types import SimpleNamespace

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from scipy.spatial.distance import cdist
from scipy.special import softmax

import json
import numpy as np
import warnings
import pickle as pk
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd  # or json, csv, pickle …
import numpy as np, torch, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
from pathlib import Path
from sklearn.manifold import TSNE

from .dataset import prepare_datasets, collate_fn_padd
from ..model import *

from ..specaug import SpecAug
from ..noise import AddNoise

warnings.filterwarnings("ignore")


def class_balanced_softmax_cross_entropy_with_softtarget(inputs, targets, weights, reduction='mean'):
    """
    Retains the original class_balanced loss logic.
    """
    #print(f"weight bef: {weights.shape}")
    weights = (weights.unsqueeze(0).repeat(targets.size(0), 1) * targets).sum(dim=1, keepdim=True)
    #print(f"weights aft: {weights.shape} {weights}")
    log_probs = F.log_softmax(inputs.view(inputs.size(0), -1), dim=1)
    loss = -(weights * targets.view(targets.size(0), -1) * log_probs).sum(dim=1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def softmax_cross_entropy_with_softtarget(inputs, targets, reduction='mean'):
    """
    A standard softmax cross-entropy loss with soft targets (no class-balancing).
    """
    log_probs = F.log_softmax(inputs.view(inputs.size(0), -1), dim=1)
    loss = -(targets.view(targets.size(0), -1) * log_probs).sum(dim=1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class DownstreamExpert(nn.Module):
    """
    Modified DownstreamExpert to incorporate:
      1) Teacher–Student knowledge distillation
      2) EMA updates of teacher
      3) Pseudo-label generation
    """

    def __init__(self, upstream_dim, runner, downstream_expert, expdir,
                 ema_decay=0.999, confidence_threshold=0.0, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir
        self.total_steps = runner['total_steps']

        # For knowledge distillation
        self.ema_decay = ema_decay
        self.confidence_threshold = confidence_threshold
        # self.warmup_step = 144 #144 #30
        # max_iter = args.max_epoch * len(dset_loaders["target"])
        # self.interval_iter = 144 #max_iter // args.interval

        # Identify test fold
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant", "fold1")
        print(f"[Expert] - Using testing fold: \"{self.fold}\".")

        # Prepare datasets
        if self.datarc['corpus'] == 'CREMA-D' or self.datarc['corpus'] == 'PODCAST' or self.datarc['corpus'] == 'IEMOCAP':
            (self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
            self.class_balanced_weights,
            self.k_thresold,
            self.all_emotions) = prepare_datasets(
                self.datarc,
                self.datarc['root'] + self.datarc['corpus'] + '/' + self.datarc['p_or_s'] + '/' + self.datarc['src'] + "/config.json"
            )
            config_path = os.path.join(self.datarc['root'], self.datarc['corpus'], self.datarc['p_or_s'], self.datarc['src'], "config.json") 
        else:
            (self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
            self.class_balanced_weights,
            self.k_thresold,
            self.all_emotions) = prepare_datasets(
                self.datarc,
                self.datarc['root'] + self.datarc['corpus'] + '/' + self.datarc['p_or_s'] + "/config.json" #+ self.datarc['src']
            )
            config_path = os.path.join(self.datarc['root'], self.datarc['corpus'], self.datarc['p_or_s'], "config.json") #, self.datarc['src']


        # Load config with label information
        # config_path = os.path.join(self.datarc['root'], self.datarc['corpus'], self.datarc['p_or_s'], "config.json") #, self.datarc['src']
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Model definition (student & teacher)
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})

        # model
        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=len(self.config['categorical']["emo_type"]),
            **model_conf
        )

        # # Teacher model
        # self.teacher_model = model_cls(
        #     input_dim=self.modelrc['projector_dim'],
        #     output_dim=len(self.config['categorical']["emo_type"]),
        #     **model_conf
        # )
        # self._initialize_teacher()

        # Loss function (class balanced with soft targets)
        self.objective = class_balanced_softmax_cross_entropy_with_softtarget

        # Logging
        self.register_buffer('best_score', torch.ones(1) * 99999)
        self.best_dev_loss = torch.ones(1) * 99999
        self.best_dev_acc = torch.tensor(0.0)
        # Augmentations
        # self.teacher_augmentation = AddNoise(noise_mean=0.0, noise_std=0.005, intensity=1.0)
        self.augmentation = SpecAug(
            apply_time_warp=True,
            time_warp_window=5,
            apply_freq_mask=True,
            freq_mask_width_range=(0, 20),
            apply_time_mask=True,
            time_mask_width_range=(0, 100),
        )
        self.mem_label = None
    # ------------------------------------------------
    # Teacher Model (EMA) Functions
    # ------------------------------------------------
    def _initialize_teacher(self):
        """
        Initialize the teacher model with the student's weights.
        """
        for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            teacher_param.data.copy_(student_param.data)
            teacher_param.requires_grad = False  # Freeze teacher

    def _ema_update_teacher(self):
        """
        Apply EMA update to the teacher model's weights using the student's weights.
        """
        for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            teacher_param.data = self.ema_decay * teacher_param.data + (1.0 - self.ema_decay) * student_param.data

    # ------------------------------------------------
    # Dataloader Interface
    # ------------------------------------------------
    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn_padd
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc['eval_batch_size'],
            shuffle=False,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn_padd
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def get_dataloader(self, mode):
        return getattr(self, f'get_{mode}_dataloader')()

    # ------------------------------------------------
    # Pseudo-label Generation (Teacher)
    # ------------------------------------------------
    def generate_pseudo_labels(self, upstream_model, featurizer_model, projector, classifier, args):
        """
        Generate pseudo-labels via teacher model + Gaussian noise augmentation.
        Returns:
          pseudo_labels: [batch] of label indices (argmax) or distribution
          mask: Boolean mask for samples above confidence threshold
        """
        # args: args.device, args.epsilon, args.class_num, args.distance, args.threshold, args.out_file
        # (loader, netF, netB, netC, args):
        loader = self.get_train_dataloader()
        start_test = True
        # all_id = []
        # with torch.cuda.amp.autocast(enabled=amp):   
        with torch.no_grad():
            for batch_id, (wavs, labels, filenames) in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=False):
                    wavs = [torch.FloatTensor(wav).to(args.device) for wav in wavs]
                    h_features = upstream_model(wavs)
                    feas = featurizer_model(wavs, h_features)
                    # print(f"after featurizer: {len(feas)}{feas[0].shape}")
                    features_len = torch.IntTensor([len(feat) for feat in feas]).to(device=args.device)
                    # labels: assumed to be soft labels with shape (batch_size, num_classes)
                    # features: [batch, feat_dim]
                    feas = pad_sequence(feas, batch_first=True)
                    # print(f"after pad_sequence: {feas.shape}")
                    feas = projector(feas)
                    # print(f"after projector: {feas.shape}")
                    logits, pooled, _ = classifier(feas, features_len)      # [batch, num_classes]
                    # print(f"logits: {len(logits[0])}")
                # print(f"filenames: {filenames}")
                if start_test:
                    all_fea = pooled.float().cpu()
                    all_output = logits.float().cpu()
                    all_label = labels.float().cpu()
                    all_id = filenames
                    # print(f"all_id: {all_id}")
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, pooled.float().cpu()), dim=0)
                    all_output = torch.cat((all_output, logits.float().cpu()), dim=0)
                    all_label = torch.cat((all_label, labels.float().cpu()), dim=0)
                    # print(f"all_id: {batch_id} {all_id}")
                    all_id.extend(filenames) #torch.cat((all_id, filenames), dim=0)
                

        # Apply softmax to outputs
        all_output = nn.Softmax(dim=1)(all_output)  # [N, C]
        print(f"all_output: {all_output.shape}")
        print(f"all_fea: {all_fea.shape}")
        print(f"all_label: {all_label.shape}")
        # Compute entropy and confidence
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
        unknown_weight = 1 - ent / np.log(args.class_num)

        if args.distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), dim=1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy()        # [N, feat_dim]
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()         # [N, C]
        # all_label = all_label.float().cpu().numpy()                  # [N, C]
        # all_id = all_id.numpy()
        
        for _ in range(2):
            # Soft centroid computation
            initc = aff.T @ all_fea                   # [C, feat_dim]
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  # normalize

            # Only keep classes with enough pseudo-labels
            cls_count = aff.sum(axis=0)               # [C]
            labelset = np.where(cls_count > args.threshold)[0]

            # Distance to selected centroids
            dd = cdist(all_fea, initc[labelset], args.distance)  # [N, valid_C]

            # Soft pseudo-labels using negative distances
            soft_pseudo = softmax(-dd, axis=1)        # [N, valid_C]

            # Reassign to full class dimension
            aff = np.zeros((all_fea.shape[0], K))     # [N, C]
            aff[:, labelset] = soft_pseudo            # update only valid classes

        # Macro F1 
        # Summarize the prediction distribution
        prediction_distribution = torch.from_numpy(aff)
        predictions_binary = torch.where(prediction_distribution > self.k_thresold, 1.0, 0.0)
        labels_binary = torch.where(all_label > self.k_thresold, 1.0, 0.0)

        all_emotions = self.all_emotions
        reprot_dict = classification_report(
            labels_binary.cpu().numpy(force=True),
            predictions_binary.cpu().numpy(force=True),
            target_names=all_emotions,
            output_dict=True
        )
        macro_f1 = reprot_dict['macro avg']['f1-score']
        log_str = f'AccuracyL macro f1 = {macro_f1 * 100:.2f}% (soft target match)'
        
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str + '\n')
        
        soft_pl_dict = dict()
        for i, soft_pl in enumerate(aff):
            filename = all_id[i]
            soft_pl_dict[filename] = soft_pl
            
        return soft_pl_dict #aff  # soft pseudo-labels, shape [N, C]
    
    # ------------------------------------------------
    # dynamic class balanced weighting
    # ------------------------------------------------
    def compute_class_weights_from_probs_thresholded(
        self,
        teacher_probs: torch.Tensor,
        method: str = 'log-inverse',
        threshold: float = None,
        beta: float = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute adaptive class weights from soft teacher probabilities,
        using a thresholded multi-label selection method.

        Args:
            teacher_probs (Tensor): [B, C] softmax outputs from teacher.
            method (str): One of {'log-inverse', 'effective-num'}.
            threshold (float): If None, defaults to 1 / num_classes.
            beta (float): For effective number method. If None, defaults to (B - 1)/B.
            eps (float): Small value for numerical stability.

        Returns:
            class_weights (Tensor): [C] normalized class weights.
        """
        assert method in {'log-inverse', 'effective-num'}, "Invalid method selected."

        B, C = teacher_probs.shape
        device = teacher_probs.device

        if threshold is None:
            threshold = 1.0 / C

        # Step 1: Binary multi-label mask based on threshold
        multi_label_mask = (teacher_probs > threshold).float()  # [B, C]

        # Step 2: Count selected samples per class
        alpha_c = multi_label_mask.sum(dim=0)  # shape: [C]

        # Step 3: Weighting schemes
        if method == 'log-inverse':
            max_alpha = alpha_c.max().clamp(min=1.0)
            freq_ratio = alpha_c / max_alpha
            weights = 1.0 - torch.log(freq_ratio + eps)

        elif method == 'effective-num':
            if beta is None:
                beta = (B - 1) / B
            effective_num = 1.0 - torch.pow(beta, alpha_c)
            weights = (1.0 - beta) / (effective_num + eps)

        # Step 4: Normalize to sum C (like your original class_balanced_weights)
        class_weights = weights / weights.sum() * C
        return class_weights.detach()

    def plot_feature_kde(self, pooled: torch.Tensor, gt: torch.Tensor, save_prefix: str = 'dev_feats', n_grid: int = 200, cmap: str = 'jet', reducer: str = 'umap', per_class = None):
        try:
            import umap
            HAS_UMAP = True
        except ImportError:
            HAS_UMAP = False

        X = pooled.float().numpy()
        y = gt.numpy()

        # -------- balanced subsample ----------------------------------
        if per_class is not None:
            idx_keep = []
            for c in np.unique(y):
                idx_c = np.where(y == c)[0]
                idx_keep.extend(np.random.choice(idx_c,
                                size=min(per_class, len(idx_c)),
                                replace=False))
            idx_keep = np.array(idx_keep)
            X = X[idx_keep]
            y = y[idx_keep]

        # -------- dimension reduction ---------------------------------
        if reducer == 'pca':
            Z = PCA(n_components=2, whiten=False, random_state=0).fit_transform(X)
        elif reducer == 'tsne':
            Z = TSNE(n_components=2, perplexity=30,
                    init='pca', random_state=0).fit_transform(X)
        else:                            # UMAP
            assert HAS_UMAP, "pip install umap-learn"
            Z = umap.UMAP(n_components=2, random_state=0,
                        metric='cosine').fit_transform(X)

        # -------- scatter plot ----------------------------------------
        plt.figure(figsize=(4,4), dpi=160)
        plt.scatter(Z[:,0], Z[:,1], c=y, cmap='tab10', s=6, alpha=.7)
        plt.xticks([]); plt.yticks([]); plt.tight_layout()
        plt.title(f'{reducer.upper()} scatter')
        scatter_path = Path(self.expdir) / f'{save_prefix}_scatter.png'
        plt.savefig(scatter_path, dpi=160); plt.close()

        # -------- KDE heat-map ----------------------------------------
        kde  = gaussian_kde(Z.T, bw_method='scott')
        xmin, ymin = Z.min(0)-.05
        xmax, ymax = Z.max(0)+.05
        xx, yy = np.mgrid[xmin:xmax:complex(n_grid),
                        ymin:ymax:complex(n_grid)]
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        plt.figure(figsize=(4,4), dpi=160)
        plt.imshow(zz.T, origin='lower', extent=[xmin,xmax,ymin,ymax], cmap=cmap)
        plt.xticks([]); plt.yticks([]); plt.tight_layout()
        plt.title(f'KDE heat-map ({reducer})')
        scatter_path = Path(self.expdir) / f'{save_prefix}_{reducer}_kde2d.png'
        plt.savefig(scatter_path, dpi=160); plt.close()

        # -------- 3-D surface (optional) ------------------------------
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        fig = plt.figure(figsize=(5,4), dpi=160)
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, zz, cmap=cmap,
                        rcount=80, ccount=80,
                        linewidth=0, antialiased=False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((1,1,0.25)); ax.view_init(30,45)
        plt.tight_layout()
        scatter_path = Path(self.expdir) / f'{save_prefix}_{reducer}_kde3d.png'
        plt.savefig(scatter_path, dpi=160); plt.close()

    
    # ------------------------------------------------
    # Forward (Student + Distillation)
    # ------------------------------------------------
    def forward(self, mode, features, labels, filenames, tar_idx, records, global_step, args, fea_bank=None, score_bank=None, vis_feats=[], vis_labs=[], **kwargs):
        """
        Single forward pass. 
        Modified to incorporate:
          1) Pseudo-label generation
          2) Distillation loss
          3) EMA update of teacher
        """
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        
        features = pad_sequence(features, batch_first=True)
        tar_idx = torch.tensor(tar_idx)
        #print(f"after pad_sequence: {features.shape}")

        # === Modified / New Lines: Distillation for 'train' mode only === #
        if mode == 'train':
            features = self.projector(features)
            logits, pooled, _ = self.model(features, features_len)
            softmax_out = nn.Softmax(dim=1)(logits)
            #output_re = softmax_out.unsqueeze(1)

            with torch.no_grad():
                output_f_norm = F.normalize(pooled)
                output_f_ = output_f_norm.cpu().detach().clone()

                fea_bank[tar_idx] = output_f_.detach().clone().cpu()
                score_bank[tar_idx] = softmax_out.detach().clone()

                distance = output_f_@fea_bank.T #batch x n
                _, idx_near = torch.topk(distance,
                                        dim=-1,
                                        largest=True,
                                        k=args.K+1) ## define args.K, args.NRC
                idx_near = idx_near[:, 1:]  #batch x K
                score_near = score_bank[idx_near]    #batch x K x C

                fea_near = fea_bank[idx_near]  #batch x K x num_dim
                fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
                distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
                _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=args.KK+1)  # M near neighbors for each of above K ones
                idx_near_near = idx_near_near[:,:,1:] # batch x K x M
                tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
                match = (
                    idx_near_near == tar_idx_).sum(-1).float()  # batch x K
                weight = torch.where(
                    match > 0., match,
                    torch.ones_like(match).fill_(0.1))  # batch x K

                weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                        args.KK)  # batch x K x M
                weight_kk = weight_kk.fill_(0.1)

                # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
                #weight_kk[idx_near_near == tar_idx_]=0

                score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                #print(weight_kk.shape)
                weight_kk = weight_kk.contiguous().view(weight_kk.shape[0], -1)  # batch x KM

                score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1, len(self.all_emotions))  # batch x KM x C

                score_self = score_bank[tar_idx]

            if args.class_weights:            # <-- True  ➜  use adaptive weights
                class_weights = self.compute_class_weights_from_probs_thresholded(
                    torch.softmax(logits, dim=-1),  # shape: (batch_size, num_classes),
                    method='effective-num',  # or 'effective-num', # 'log-inverse'
                    threshold=1.0 / softmax_out.shape[1]  # optional
                )
            else:                            # <-- False ➜ uniform weighting
                class_weights = logits.new_ones(logits.size(-1))   # (C,)
                print(f"shape of class_weights: {class_weights.shape}")
            
            # class_weights = self.compute_class_weights_from_probs_thresholded(
            #     torch.softmax(logits, dim=-1),  # shape: (batch_size, num_classes),
            #     method='effective-num',  # or 'effective-num', # 'log-inverse'
            #     threshold=1.0 / softmax_out.shape[1]  # optional
            # )
            # nn of nn
            output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK, -1)  # batch x K*KK x C
            const = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) * weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
            near_kk_loss = torch.mean(const)
            near_kk_loss *= args.lambda_nearkk
            # nn
            softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C

            near_k_loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))
            near_k_loss *= args.lambda_neark
            # self, if not explicitly removing the self feature in expanded neighbor then no need for this
            #loss += -torch.mean((softmax_out * score_self).sum(-1))
            # reg_loss = -torch.mean((softmax_out * score_self).sum(-1))

            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(class_weights* msoftmax * torch.log(msoftmax + 1e-5)) # args.epsilon))
            gentropy_loss *= args.lambda_div
            # -------------------------------
            # *Optimization: Final Loss
            # -------------------------------
            final_loss = near_kk_loss + near_k_loss + gentropy_loss # + reg_loss
            
            
            records['loss'].append(final_loss.item()) #distill_loss.item()) #
            records['near_k_loss'].append(near_k_loss.item())
            records['near_kk_loss'].append(near_kk_loss.item())
            # records['reg_loss'].append(reg_loss.item())
            records['gentropy_loss'].append(gentropy_loss.item())
            
            records['acc'].append(0.0)  # There's no direct "acc" on unlabeled data
            records["filename"] += filenames
            # For simplicity, no predictions/truth logging in pseudo-label scenario
            records["predict"] += ["pseudo"] * len(filenames)
            records["truth"] += ["unlabeled"] * len(filenames)

            return final_loss

        else:
            # === Original code path for dev/test (supervised scenario) === #

            features = self.projector(features)
            predicted, pooled, _ = self.model(features, features_len)
            if global_step in {1, int(self.total_steps*0.05), int(self.total_steps*0.5), self.total_steps} and mode=="dev":
                gt_dev = torch.argmax(labels.to(features.device), dim=1)
                # print(f"dev step {global_step} pooled: {pooled.shape}, gt_dev: {gt_dev.shape}") 
                vis_feats.append(pooled.cpu())
                vis_labs.append(gt_dev.cpu())
            labels = labels.to(features.device)

            # Compute the normal supervised loss
            if args.class_weights:
                class_weights = self.compute_class_weights_from_probs_thresholded(
                    torch.softmax(predicted, dim=-1),
                    method='effective-num',
                    threshold=1.0 / softmax_out.shape[1]
                )
            else:
                class_weights = predicted.new_ones(predicted.size(-1))
            
            loss = self.objective(predicted, labels, class_weights.to(device), reduction='mean')
            
            predicted_class = torch.argmax(predicted, dim=1)
            true_class = torch.argmax(labels, dim=1)
            # For classification report (CPU numpy arrays)
            predicted_np = predicted_class.cpu().numpy(force=True)
            labels_np = true_class.cpu().numpy(force=True)
            
            # Summarize the prediction distribution
            predictions_binary = torch.zeros_like(predicted, dtype=torch.float)  # (B, C)
            predictions_binary.scatter_(1, predicted_class.unsqueeze(1), 1.0)

            labels_binary = torch.zeros_like(labels, dtype=torch.float)          # (B, C)
            labels_binary.scatter_(1, true_class.unsqueeze(1), 1.0)
            # prediction_distribution = torch.nn.functional.softmax(predicted, dim=1)
            # predictions_binary = torch.where(prediction_distribution > self.k_thresold, 1.0, 0.0)
            # labels_binary = torch.where(labels > self.k_thresold, 1.0, 0.0)

            all_emotions = self.all_emotions
            reprot_dict = classification_report(
                labels_binary.cpu().numpy(force=True),
                predictions_binary.cpu().numpy(force=True),
                target_names=all_emotions,
                output_dict=True
            )
            hamming_acc = 1.0 - torch.logical_xor(predictions_binary, labels_binary).float().mean().item()
            # hamming_acc = 1.0 - (predictions_binary ^ labels_binary).float().mean().item()
            records['ham_acc'] += [hamming_acc]
            records['acc'] += [accuracy_score(labels_np, predicted_np)]
            records['precision'] += [reprot_dict['macro avg']['precision']]
            records['recall'] += [reprot_dict['macro avg']['recall']]
            records['macro-f1'] += [reprot_dict['macro avg']['f1-score']]
            # records['acc'] += [reprot_dict['macro avg']['f1-score']]
            records['loss'].append(loss.item())
            records["filename"] += filenames

            # Generate readable prediction/label strings
            all_emotions_np = np.array(all_emotions)
            predict_strs = []
            truth_strs = []
            for idx in range(len(labels_binary)):
                truth_emo = ";".join(list(all_emotions_np[np.where(labels_binary[idx].cpu().numpy(force=True) == 1.0)[0]]))
                predict_emo = ";".join(list(all_emotions_np[np.where(predictions_binary[idx].cpu().numpy(force=True) == 1.0)[0]]))
                predict_strs.append(predict_emo)
                truth_strs.append(truth_emo)

            records["predict"] += predict_strs
            records["truth"] += truth_strs

            return loss

    # ------------------------------------------------
    # log_records
    # ------------------------------------------------
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []

        # log acc & loss
        for key in ["acc", "precision", "recall", "macro-f1", "ham_acc", "loss", "near_k_loss", "near_kk_loss", "gentropy_loss"]: # , "reg_loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()

            logger.add_scalar(
                f'emotion-{self.fold}/{mode}-{key}',
                average,
                global_step=global_step
            )

            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'loss':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} {key} at step {global_step}: {average}\n')
                    if mode == 'dev' and average < self.best_dev_loss: #self.best_score:
                        # self.best_score = torch.ones(1) * average
                        self.best_dev_loss = torch.ones(1) * average
                        f.write(f'New best on {mode} {key} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')
                elif key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} {key} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_dev_acc: #self.best_score:
                        # self.best_score = torch.ones(1) * average
                        self.best_dev_acc = torch.ones(1) * average
                        f.write(f'New best on {mode} {key} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')
                elif key == 'precision' or key == 'recall' or key == 'macro-f1':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} {key} at step {global_step}: {average}\n')
                elif key == 'ham_acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} {key} at step {global_step}: {average}\n')    

        # Write out predictions
        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                lines = [f"{fn} {pred}\n" for fn, pred in zip(records["filename"], records["predict"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                lines = [f"{fn} {tr}\n" for fn, tr in zip(records["filename"], records["truth"])]
                file.writelines(lines)

        return save_names
