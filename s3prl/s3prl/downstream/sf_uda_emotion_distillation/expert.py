import os
import math
import torch
import random
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

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
        self.warmup_step = 144 #144 #30

        # Identify test fold
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant", "fold1")
        print(f"[Expert] - Using testing fold: \"{self.fold}\".")

        # Prepare datasets
        if self.datarc['corpus'] == 'CREMA-D' or self.datarc['corpus'] == 'PODCAST' or self.datarc['corpus'] == "IEMOCAP":
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
        self.student_projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.teacher_projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})

        # Student model
        self.student_model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=len(self.config['categorical']["emo_type"]),
            **model_conf
        )

        # Teacher model
        self.teacher_model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=len(self.config['categorical']["emo_type"]),
            **model_conf
        )
        self._initialize_teacher()

        # Loss function (class balanced with soft targets)
        self.objective = class_balanced_softmax_cross_entropy_with_softtarget

        # Logging
        self.register_buffer('best_score', torch.ones(1) * 99999)
        self.best_dev_loss = torch.ones(1) * 99999
        self.best_dev_acc = torch.tensor(0.0)
        # Augmentations
        self.teacher_augmentation = AddNoise(noise_mean=0.0, noise_std=0.005, intensity=1.0)
        self.student_augmentation = SpecAug(
            apply_time_warp=True,
            time_warp_window=5,
            apply_freq_mask=True,
            freq_mask_width_range=(0, 20),
            apply_time_mask=True,
            time_mask_width_range=(0, 100),
        )

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
        
        for teacher_proj_param, student_proj_param in zip(self.teacher_projector.parameters(), self.student_projector.parameters()):
            teacher_proj_param.data = self.ema_decay * teacher_proj_param.data + (1.0 - self.ema_decay) * student_proj_param.data

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
    def generate_pseudo_labels(self, features):
        """
        Generate pseudo-labels via teacher model + Gaussian noise augmentation.
        Returns:
          pseudo_labels: [batch] of label indices (argmax) or distribution
          mask: Boolean mask for samples above confidence threshold
        """
        with torch.no_grad():
            features, x_lengths = self.teacher_augmentation(features)
            """
            if x_lengths is None:
                x_lengths = torch.LongTensor([x.size(0) for x in features])
            batchsize, max_len, dim = len(x_lengths), torch.max(x_lengths).item(), features[0].size(1)

            # Pad sequences to the same length
            features_pad = features[0].new_zeros((batchsize, max_len, dim))
            for i, x in enumerate(xs):
                xs_pad[i, :x_lengths[i]] = x
            """
            features = pad_sequence(features, batch_first=True)
            features = self.teacher_projector(features)
            logits, _, _ = self.teacher_model(features, None)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, pseudo_labels = probabilities.max(dim=-1)
            # print(f"pseudo_labels: {pseudo_labels.shape}")
            # print(f"confidence: {confidence.shape}")

        # Filter out low-confidence samples
        mask = confidence >= self.confidence_threshold
        return pseudo_labels, mask, probabilities

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
    def forward(self, mode, features, labels, filenames, records, addi_features, global_step, args, vis_feats=[], vis_labs=[], **kwargs):
        """
        Single forward pass. 
        Modified to incorporate:
          1) Pseudo-label generation
          2) Distillation loss
          3) EMA update of teacher
        """
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        # initial features: list of each sample(seq_len, dim) with its own feature length
        # Pad input to same length
        features = pad_sequence(features, batch_first=True)
        #print(f"after pad_sequence: {features.shape}")

        # === Modified / New Lines: Distillation for 'train' mode only === #
        if mode == 'train':
            # -------------------------------------------------------
            #  1) Generate pseudo-labels from teacher
            # -------------------------------------------------------
            with torch.no_grad():
                # For unlabeled or target domain adaptation scenario:
                # If you want to combine real labels + pseudo labels, adapt this logic.
                pseudo_labels, mask, teacher_probs = self.generate_pseudo_labels(addi_features) #features)

            # -------------------------------------------------------
            #  2) Student model forward pass & Distillation Loss
            # -------------------------------------------------------
            # Optionally augment the student input differently from the teacher
            features_aug, _ = self.student_augmentation(features)
            features_aug = pad_sequence(features_aug, batch_first=True)
            
            features_aug = self.student_projector(features_aug)

            # Student predictions
            student_logits, _, _ = self.student_model(features_aug, features_len)
            #print(f"logits shape: {student_logits.shape}")

            # We'll build "soft targets" from teacher_probs or "hard targets" from pseudo_labels
            # If you prefer soft targets:
            teacher_soft_targets = teacher_probs  # shape [batch, #classes]

            # Only keep samples above threshold
            if global_step==1:
                print(f"pseudo label confidence mask: {mask.shape}")
            selected_logits = student_logits[mask]
            selected_teacher_probs = teacher_soft_targets[mask]
            
            if args.class_weights:            # <-- True  ➜  use adaptive weights
                class_weights = self.compute_class_weights_from_probs_thresholded(
                    teacher_probs,  # shape: (batch_size, num_classes),
                    method='effective-num',  # or 'effective-num', # 'log-inverse'
                    threshold=1.0 / teacher_probs.shape[1]  # optional
                )
            else:                            # <-- False ➜ uniform weighting
                class_weights = teacher_probs.new_ones(teacher_probs.size(-1))   # (C,)
                print(f"shape of class_weights: {class_weights.shape}")
                
            # class_weights = self.compute_class_weights_from_probs_thresholded(
            #     teacher_probs,
            #     method='effective-num',  # or 'effective-num', # 'log-inverse'
            #     threshold=1.0 / teacher_probs.shape[1]  # optional
            # )
            # EMA distillation loss: Consistency Loss + Pseudo Labeling 
            if selected_logits.shape[0] == 0:
                # If no sample is above threshold, fallback to a zero-loss
                distill_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                distill_loss = self.objective(
                    selected_logits,
                    selected_teacher_probs,      # soft targets
                    class_weights.to(device), # self.class_balanced_weights.to(device), # 
                    reduction='mean'
                )
                print(f"distill_loss: {distill_loss}")
            
            ent = False #True
            fbnm = True
            # -------------------------------
            # entropy minimization loss
            # -------------------------------
            if ent:
                student_probs = torch.softmax(student_logits, dim=-1)  # shape: (batch_size, num_classes)
                student_log_probs = torch.log_softmax(student_logits, dim=-1)  # shape: (batch_size, num_classes)
                entropy = -torch.sum(class_weights * student_probs * student_log_probs, dim=-1)  # shape: (batch_size,)
                ent_loss = entropy.mean(dim=0)
                print(f"ent_loss: {ent_loss}")
            else:
                ent_loss = torch.tensor(0.0, device=device, requires_grad=True)

            lambda_ent = args.lambda_ent # 0.1
            
            # -------------------------------
            # diversity loss
            # -------------------------------
            # final_loss = distill_loss
            student_probs = torch.softmax(student_logits, dim=-1)   
            mean_probs = student_probs.mean(dim=0)
            div_loss = (class_weights * mean_probs * torch.log(mean_probs + 1e-6)).sum()
            print(f"div_loss: {div_loss}")
            lambda_div = args.lambda_div # 0.1
            
            # if global_step < self.warmup_step:
            #     final_loss = distill_loss
            # else:
            #     lambda_div = 0.1
            #     final_loss = distill_loss + lambda_div * div_loss
            
            # -------------------------------
            # *Nuclear Norm Loss (Frobenius)
            # -------------------------------
            if fbnm:
                softmax_out = torch.softmax(student_logits, dim=-1) 
                list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_out,2),dim=0)), descending=True)
                fbnm_loss = - torch.mean(list_svd[:min(softmax_out.shape[0],softmax_out.shape[1])])
                #fbnm_loss = lambda_fbnm*fbnm_loss
                print(f"fbnm_loss: {fbnm_loss}")
            else:
                fbnm_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
            lambda_fbnm = args.lambda_fbnm # 0.1
            
            # student_probs_all = torch.softmax(student_logits, dim=1)
            # nuclear_norm_loss = -torch.norm(student_probs_all, p='fro')  # encourage diversity
            # print(f"distill: {distill_loss}, nuclear_norm: {nuclear_norm_loss}; {nuclear_norm_loss*0.2}")
            # Weight for nuclear norm term
            
            # if step < warmup_step:
            #     final_loss = distill_loss
            # else:
            #     lambda_nm = 0.01
            #     final_loss = distill_loss + lambda_nm * nuclear_norm_loss
            src_tag = torch.full((selected_teacher_probs.size(0),), 0, dtype=torch.int8, device=selected_teacher_probs.device)
            pseudo_class = torch.argmax(selected_teacher_probs, dim=1) #pseudo_label, dim=1)
            labels = labels.to(features.device)
            true_class = torch.argmax(labels, dim=1)
            log_path = Path(self.expdir) / "pseudo_diagnostics.tsv"
            header   = "\t".join(
                ["step","file","gt","src","correct",
                ]) + "\n"

            if global_step == 1 and not log_path.exists():
                log_path.write_text(header)            # create header once

            with log_path.open("a") as lf:
                for i, fn in enumerate(filenames):
                    correct = int(pseudo_class[i] == true_class[i])
                    
                    row = [
                        f"{global_step}",
                        fn,
                        f"{true_class[i]}",
                        f"{src_tag[i].item()}",
                        f"{correct}",
                    ]
                    lf.write("\t".join(row) + "\n")
            
            # -------------------------------
            # *Optimization: Final Loss
            # -------------------------------
            final_loss = distill_loss + lambda_div * div_loss + lambda_ent * ent_loss + lambda_fbnm * fbnm_loss
            # -------------------------------------------------------
            #  3) EMA update of teacher after backward
            # -------------------------------------------------------
            # The typical pattern is to return distill_loss,
            # then do `distill_loss.backward()`, then `_ema_update_teacher()`.
            # If your trainer calls `forward()` + `backward()` each step,
            # you'd do `_ema_update_teacher()` AFTER the backward outside here.
            # 
            # However, if you want to do it within forward for demonstration:
            # (Note: this is less typical; in practice you'd do it in your training loop.)
            # We show it here for clarity:
            #
            self._ema_update_teacher()
            #
            # We'll NOT do it here automatically to avoid messing up your existing trainer logic.

            # Collect logging info (loss, predictions) similar to your original code
            # We skip real label usage here because we're focusing on unlabeled distillation.
            # If you also have supervised labels, you can combine them:
            #   final_loss = distill_loss + alpha * supervised_loss
            #   return final_loss
            # For now, just log distill_loss in records:
            records['loss'].append(final_loss.item()) #distill_loss.item()) #
            records['distill'].append(distill_loss.item())
            records['entropy'].append(ent_loss.item())
            records['diversity'].append(div_loss.item())
            records['fbnm'].append(fbnm_loss.item())
            # records['nuclear_norm'].append(nuclear_norm_loss.item())
            # records['loss'].append(distill_loss.item())
            records['acc'].append(0.0)  # There's no direct "acc" on unlabeled data
            records["filename"] += filenames
            # For simplicity, no predictions/truth logging in pseudo-label scenario
            records["predict"] += ["pseudo"] * len(filenames)
            records["truth"] += ["unlabeled"] * len(filenames)

            return final_loss #distill_loss #

        else:
            # === Original code path for dev/test (supervised scenario) === #

            features = self.student_projector(features)
            predicted, pooled, _ = self.student_model(features, features_len)
            if global_step in {1, int(self.total_steps*0.05), int(self.total_steps*0.5), self.total_steps} and mode=="dev":
                gt_dev = torch.argmax(labels.to(features.device), dim=1)
                # print(f"dev step {global_step} pooled: {pooled.shape}, gt_dev: {gt_dev.shape}") 
                vis_feats.append(pooled.cpu())
                vis_labs.append(gt_dev.cpu())
            labels = labels.to(features.device)
            
            with torch.no_grad():
                # For unlabeled or target domain adaptation scenario:
                # If you want to combine real labels + pseudo labels, adapt this logic.
                pseudo_labels, mask, teacher_probs = self.generate_pseudo_labels(addi_features) #features)

            class_weights = self.compute_class_weights_from_probs_thresholded(
                teacher_probs,
                method='effective-num',  # or 'effective-num', # 'log-inverse'
                threshold=1.0 / teacher_probs.shape[1]  # optional
            )

            # Compute the normal supervised loss
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
        for key in ["acc", "precision", "recall", "macro-f1", "ham_acc", "loss", "distill", "diversity", "entropy", "fbnm"]: #"nuclear_norm"]:
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
                    if mode == 'dev' and average < self.best_dev_loss: # self.best_score:
                        # self.best_score = torch.ones(1) * average
                        self.best_dev_loss = torch.ones(1) * average
                        f.write(f'New best on {mode} {key} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')
                elif key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} {key} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_dev_acc: # self.best_score:
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

'''
import torch
import torch.nn.functional as F

# ----------------------------------------
# Utility: Curriculum + ACR weight masking
# ----------------------------------------
def compute_difficulty(prob_ema, prob_proxy):
    entropy = -(prob_ema * torch.log(prob_ema + 1e-6)).sum(dim=-1)
    agreement = F.cosine_similarity(prob_ema, prob_proxy, dim=-1)
    return entropy - agreement  # Lower = easier

def dynamic_curriculum_mask(difficulty, pct):
    threshold = torch.quantile(difficulty, pct)
    return (difficulty <= threshold)  # Boolean mask

def apply_acr_weights(sample_weights, epoch, total_epochs, eta=3.0):
    e_frac = epoch / total_epochs
    pinv = 1 - torch.exp(torch.tensor(-eta * e_frac))
    flip_mask = torch.rand_like(sample_weights) < pinv
    inverted_weights = torch.where(flip_mask, 1.0 - sample_weights, sample_weights)
    return inverted_weights

# ----------------------------------------
# Integration into training forward
# ----------------------------------------
# In DownstreamExpert.forward(), inside mode == 'train':
# (Insert below where distill_loss is computed)

# Step 1: MatchOrConf Fusion (pseudo-label)
# Use Gemini and EMA teacher predictions
cos_sim = F.cosine_similarity(gemini_probs, teacher_probs, dim=-1)
entropy_gemini = -(gemini_probs * gemini_probs.log()).sum(dim=-1)
entropy_ema = -(teacher_probs * teacher_probs.log()).sum(dim=-1)

pseudo_label = torch.where(
    cos_sim > 0.9,
    0.5 * (gemini_probs + teacher_probs),
    torch.where(entropy_gemini < entropy_ema, gemini_probs, teacher_probs)
)

# Step 2: Curriculum Masking
with torch.no_grad():
    difficulty = compute_difficulty(teacher_probs, gemini_probs)
    easy_mask = dynamic_curriculum_mask(difficulty, pct=0.6)  # Example: top 60% easiest samples

# Step 3: ACR Reweighting
raw_weights = easy_mask.float()
weights = apply_acr_weights(raw_weights, epoch=global_step, total_epochs=args.total_epochs)

# Step 4: Masked distillation loss
selected_logits = student_logits[mask]
selected_targets = pseudo_label[mask]
loss_vec = F.kl_div(F.log_softmax(selected_logits, dim=-1), selected_targets, reduction='none').sum(dim=1)
distill_loss = (weights[mask] * loss_vec).mean()

# Replace previous distill_loss with this
final_loss = distill_loss + lambda_div * div_loss + lambda_ent * ent_loss + lambda_fbnm * fbnm_loss
'''