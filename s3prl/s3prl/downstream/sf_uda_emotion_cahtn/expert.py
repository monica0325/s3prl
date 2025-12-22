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

    def __init__(self, upstream_dim, downstream_expert, expdir,
                 ema_decay=0.999, confidence_threshold=0.0, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        # For knowledge distillation
        self.ema_decay = ema_decay
        self.confidence_threshold = confidence_threshold
        self.warmup_step = 144 #144 #30
        # max_iter = args.max_epoch * len(dset_loaders["target"])
        self.interval_iter = kwargs.get("interval_iter", 144) #144 #max_iter // args.interval

        # Identify test fold
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant", "fold1")
        print(f"[Expert] - Using testing fold: \"{self.fold}\".")

        # Prepare datasets
        (self.train_dataset,
         self.dev_dataset,
         self.test_dataset,
         self.class_balanced_weights,
         self.k_thresold,
         self.all_emotions) = prepare_datasets(
             self.datarc,
             self.datarc['root'] + self.datarc['corpus'] + '/' + self.datarc['p_or_s'] + '/' + self.datarc['src'] + "/config.json"
         )

        # Load config with label information
        config_path = os.path.join(self.datarc['root'], self.datarc['corpus'], self.datarc['p_or_s'], self.datarc['src'], "config.json")
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
        self.epoch_soft_labels = None
        self.epoch_neg_masks = None
        # self.all_probs = None
        # self.all_ids = None
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
    def collect_probs(self, upstream_model, featurizer_model, projector, classifier, device):
        loader = self.get_train_dataloader()
        start_test = True
        all_probs, all_ids = [], []
        
        with torch.no_grad():
            for batch_id, (wavs, labels, filenames, idxs) in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=False):
                    wavs = [torch.FloatTensor(wav).to(device) for wav in wavs]
                    h_features = upstream_model(wavs)
                    feas = featurizer_model(wavs, h_features)
                    # print(f"after featurizer: {len(feas)}{feas[0].shape}")
                    features_len = torch.IntTensor([len(feat) for feat in feas]).to(device=device)
                    # labels: assumed to be soft labels with shape (batch_size, num_classes)
                    # features: [batch, feat_dim]
                    feas = pad_sequence(feas, batch_first=True)
                    # print(f"after pad_sequence: {feas.shape}")
                    feas = projector(feas)
                    # print(f"after projector: {feas.shape}")
                    logits, pooled, _ = classifier(feas, features_len)      # [batch, num_classes]
                    # print(f"logits: {len(logits[0])}")
                    all_probs.append(torch.softmax(logits, -1).float().cpu())
                    all_ids.extend(idxs)
        
        return torch.cat(all_probs), all_ids
                
    def build_epoch_soft_labels(self, probs: torch.Tensor, idxs: list[str], alpha=0.3, neg_threshold=0.1):
        N, C = probs.shape
        soft_labels = torch.zeros_like(probs)
        neg_mask = torch.zeros_like(probs)
        
        for c in range(C):
            scores = probs[:, c]
            num_topk = max(int(alpha* N), 1)
            topk_vals, topk_idx = torch.topk(scores, num_topk)
            soft_labels[topk_idx, c] = probs[topk_idx, c]
            
            neg_mask[:, c] = (scores < neg_threshold).float()
        soft_labels_dict = {idx: soft_labels[i] for i, idx in enumerate(idxs)}
        neg_mask_dict = {idx: neg_mask[i] for i, idx in enumerate(idxs)}
        
        return soft_labels_dict, neg_mask_dict
    
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
        hamming_acc = 1.0 - torch.logical_xor(predictions_binary, labels_binary).float().mean().item()
        
        macro_f1 = reprot_dict['macro avg']['f1-score']
        
        log_str = f'AccuracyL macro f1 = {macro_f1 * 100:.2f}% (soft target match)\nhamming accuracy = {hamming_acc * 100:.2f}'
        
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

    # ------------------------------------------------
    # Forward (Student + Distillation)
    # ------------------------------------------------
    def forward(self, mode, features, labels, filenames, idxs, records, global_step, upstream_model, featurizer_model, args, **kwargs):
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
            
            if global_step % self.interval_iter == 0 or global_step==1: # and args.cls_par > 0:
                self.model.eval()
                self.projector.eval()
                upstream_model.eval()
                featurizer_model.eval()
                
                all_probs, all_idxs = self.collect_probs(upstream_model, featurizer_model, self.projector, self.model, device)
                self.epoch_soft_labels, self.epoch_neg_masks = self.build_epoch_soft_labels(all_probs, all_idxs, alpha=0.3)
                self.model.train()
                self.projector.train()
                featurizer_model.train() 
           
            features = self.projector(features)

            logits, _, _ = self.model(features, features_len)
            #print(f"logits shape: {student_logits.shape}")
            probs = torch.softmax(logits, dim=-1)
            # Retrieve from per-epoch cache
            soft_pos_labels = torch.stack([self.epoch_soft_labels[idx] for idx in idxs]).to(device)
            neg_mask = torch.stack([self.epoch_neg_masks[idx] for idx in idxs]).to(device)
            
            if args.class_weights:            
                class_weights = self.compute_class_weights_from_probs_thresholded(
                    torch.softmax(logits, dim=-1), 
                    method='effective-num',  # 'log-inverse'
                    threshold=1.0 / logits.shape[1]  # optional
                )
            else:                          
                class_weights = logits.new_ones(logits.size(-1))
                print(f"shape of class_weights: {class_weights.shape}")
            
            # Positive Learning loss/ Negative Learning loss
            pl_loss = F.kl_div(F.log_softmax(logits, dim=1), soft_pos_labels, reduction='batchmean')
            nl_loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), weight=neg_mask)
            
            ent_loss = -torch.sum(class_weights * probs * torch.log(probs + 1e-6), dim=-1).mean(dim=0) #(-probs * torch.log(probs + 1e-6)).sum(-1).mean()
            p_bar = probs.mean(0)
            div_loss = torch.sum(- class_weights * p_bar * torch.log(p_bar + 1e-5))
            # div_loss = F.kl_div(p_bar.log(), torch.full_like(p_bar, 1.0 / C), reduction='sum')
            im_loss = ent_loss + div_loss
            st_loss = pl_loss + nl_loss
            beta = args.lambda_beta
            
            # lambda_pl = args.lambda_pl # 0.3
            
            # Mutual Information: Entropy Minimization + Diversity Loss
            
            # -------------------------------
            # *Optimization: Final Loss
            # -------------------------------
            final_loss = beta * st_loss + im_loss #pl_loss + im_loss
            
            # However, if you want to do it within forward for demonstration:
            # (Note: this is less typical; in practice you'd do it in your training loop.)
            # We show it here for clarity:
            #
            # self._ema_update_teacher()
            #
            # We'll NOT do it here automatically to avoid messing up your existing trainer logic.

            # Collect logging info (loss, predictions) similar to your original code
            # We skip real label usage here because we're focusing on unlabeled distillation.
            # If you also have supervised labels, you can combine them:
            #   final_loss = distill_loss + alpha * supervised_loss
            #   return final_loss
            # For now, just log distill_loss in records:
            records['loss'].append(final_loss.item()) #distill_loss.item()) #
            records['self_training'].append(st_loss.item())
            records['information_maximization'].append(im_loss.item())
            
            records['acc'].append(0.0)  # There's no direct "acc" on unlabeled data
            records["filename"] += filenames
            # For simplicity, no predictions/truth logging in pseudo-label scenario
            records["predict"] += ["pseudo"] * len(filenames)
            records["truth"] += ["unlabeled"] * len(filenames)
            
            return final_loss #distill_loss #

        else:
            # === Original code path for dev/test (supervised scenario) === #

            features = self.projector(features)
            predicted, _, _ = self.model(features, features_len)
            labels = labels.to(features.device)

            # Compute the normal supervised loss
            loss = self.objective(predicted, labels, self.class_balanced_weights.to(device), reduction='mean')

            # Summarize the prediction distribution
            prediction_distribution = torch.nn.functional.softmax(predicted, dim=1)
            predictions_binary = torch.where(prediction_distribution > self.k_thresold, 1.0, 0.0)
            labels_binary = torch.where(labels > self.k_thresold, 1.0, 0.0)

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
            records['acc'] += [reprot_dict['macro avg']['f1-score']]
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
        for key in ["acc", "ham_acc", "loss", "self_training", "information_maximization"]:
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
                    if mode == 'dev' and average < self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} {key} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')
                elif key == 'acc':
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
