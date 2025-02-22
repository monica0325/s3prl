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

from .dataset import prepare_datasets, collate_fn_padd
from ..model import *

from ..specaug import SpecAug
from ..noise import AddNoise

warnings.filterwarnings("ignore")


def class_balanced_softmax_cross_entropy_with_softtarget(inputs, targets, weights, reduction='mean'):
    """
    Retains the original class_balanced loss logic.
    """
    print(f"weight bef: {weights.shape}")
    weights = (weights.unsqueeze(0).repeat(targets.size(0), 1) * targets).sum(dim=1, keepdim=True)
    print(f"weights aft: {weights.shape} {weights}")
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
            features = self.projector(features)
            logits, _ = self.teacher_model(features, None)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, pseudo_labels = probabilities.max(dim=-1)
            #print(f"pseudo_labels: {pseudo_labels.shape}")
            #print(f"confidence: {confidence.shape}")

        # Filter out low-confidence samples
        mask = confidence >= self.confidence_threshold
        return pseudo_labels, mask, probabilities

    # ------------------------------------------------
    # Forward (Student + Distillation)
    # ------------------------------------------------
    def forward(self, mode, features, labels, filenames, records, addi_features, **kwargs):
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
            
            features_aug = self.projector(features_aug)

            # Student predictions
            student_logits, _ = self.student_model(features_aug, features_len)
            #print(f"logits shape: {student_logits.shape}")

            # We'll build "soft targets" from teacher_probs or "hard targets" from pseudo_labels
            # If you prefer soft targets:
            teacher_soft_targets = teacher_probs  # shape [batch, #classes]

            # Only keep samples above threshold
            selected_logits = student_logits[mask]
            selected_teacher_probs = teacher_soft_targets[mask]

            if selected_logits.shape[0] == 0:
                # If no sample is above threshold, fallback to a zero-loss
                distill_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                distill_loss = self.objective(
                    selected_logits,
                    selected_teacher_probs,      # soft targets
                    self.class_balanced_weights.to(device),
                    reduction='mean'
                )

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
            records['loss'].append(distill_loss.item())
            records['acc'].append(0.0)  # There's no direct "acc" on unlabeled data
            records["filename"] += filenames
            # For simplicity, no predictions/truth logging in pseudo-label scenario
            records["predict"] += ["pseudo"] * len(filenames)
            records["truth"] += ["unlabeled"] * len(filenames)

            return distill_loss

        else:
            # === Original code path for dev/test (supervised scenario) === #

            features = self.projector(features)
            predicted, _ = self.student_model(features, features_len)
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
        for key in ["acc", "loss"]:
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

        # Write out predictions
        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                lines = [f"{fn} {pred}\n" for fn, pred in zip(records["filename"], records["predict"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                lines = [f"{fn} {tr}\n" for fn, tr in zip(records["filename"], records["truth"])]
                file.writelines(lines)

        return save_names
