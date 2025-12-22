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
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sns


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
        self.total_steps = runner['total_steps']
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir
        self.total_epochs = int(5000*32/5512) #self.total_steps*32/5512)
        self.epoch = 0

        # For knowledge distillation
        self.ema_decay = ema_decay
        self.confidence_threshold = confidence_threshold
        self.warmup_step = 144 #144 #30

        # Identify test fold
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant", "fold1")
        print(f"[Expert] - Using testing fold: \"{self.fold}\".")

        # Prepare datasets
        if self.datarc['corpus'] == 'CREMA-D' or self.datarc['corpus'] == 'PODCAST':
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

        # MI filtered dataset
        self.mi_filtered_train_dataset = self.train_dataset
        # Model definition (student & teacher)
        self.student_projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.teacher_projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        if model_conf.get('post_net'):
            post_net = model_conf['post_net']
        else:
            post_net = None
        # Student model
        if post_net:
            # Teacher model
            self.teacher_model = model_cls(
                input_dim=self.modelrc['projector_dim'],
                output_dim=len(self.config['categorical']["emo_type"]),
                **model_conf
            )
            model_conf['post_net']['MCDropFrameLevel']['p']=0.0
            self.student_model = model_cls(
                input_dim=self.modelrc['projector_dim'],
                output_dim=len(self.config['categorical']["emo_type"]),
                **model_conf
            )
        else:
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
        
        LLM_pred_path = self.datarc['llm_pred_path'] #os.path.join("/home/monica/SFUDA/baseline_eval/LALM", "soft_targets_improv_prob_output.json") #"gemini25pro_soft_targets_improv_prob_output.json") #"soft_targets_improv_prob_output.json")
        LLM_multi_pred_path = self.datarc['llm_multi_pred_path'] #os.path.join("/home/monica/SFUDA/baseline_eval/LALM", "gemini25flash_soft_targets_improv_prob_output_multi_plus1998.json") # 4149thinking0 #"gemini25flash_soft_targets_improv_prob_output_multi_5512_revise.json") #"gemini25flash_soft_targets_improv_prob_output_multi.json") #5512.json") #"gemini25flash_soft_targets_improv_prob_output_multi.json") #gemini25flash_soft_targets_improv_prob_output_multi_revise_4151
        #/home/monica/SFUDA/baseline_eval/LALM/gemini25flash_soft_targets_improv_prob_output_multi_plus1998.json
        with open(LLM_pred_path) as f:
            LLM_pred = json.load(f) # {"file_id": [p0, p1, ...], ...}
        with open(LLM_multi_pred_path) as f:
            LLM_multi_pred = json.load(f)

        # store as {file_id: torch.FloatTensor(C)}
        self.gemini_dict = {k: torch.tensor(v["pred"], dtype=torch.float32) for k, v in LLM_pred.items()} #v["smoothed"], dtype=torch.float32) for k, v in raw.items()}
        self.gemini_multi_pred_dict = {k: torch.tensor(v["preds"], dtype=torch.float32) for k, v in LLM_multi_pred.items()}
        
        num_classes = len(next(iter(self.gemini_dict.values())))
        assert num_classes == len(self.all_emotions)


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

    def get_train_dataloader(self, mi_filtered = False, ent_mean_filtered = False, ent_each_filtered = False, featurizer_model=None, upstream_model=None, args=None):
        if mi_filtered:
            self.mi_filtered_train_dataset = self.mi_based_sample_selection(featurizer_model, upstream_model, args)
            return self._get_train_dataloader(self.mi_filtered_train_dataset)
        elif ent_mean_filtered:
            self.ent_mean_filtered_train_dataset = self.ent_mean_based_sample_selection(featurizer_model, upstream_model, args)
            return self._get_train_dataloader(self.ent_mean_filtered_train_dataset)
        elif ent_each_filtered:
            self.ent_each_filtered_train_dataset = self.ent_each_based_sample_selection(featurizer_model, upstream_model, args)
            return self._get_train_dataloader(self.ent_each_filtered_train_dataset)
        else:
            return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def get_dataloader(self, mode):
        return getattr(self, f'get_{mode}_dataloader')()
    
    # ------------------------------------------------------------
    # MC-dropout + Gemini MI scan over the *whole* train set
    # ------------------------------------------------------------
    @torch.no_grad()
    def mi_based_sample_selection(self,
                                featurizer_model,
                                upstream_model,
                                args,
                                device=None,
                                T: int = 8) -> torch.utils.data.Dataset:
        """
        Returns *filtered* torch.utils.data.Subset that excludes samples whose
        MI is simultaneously in the top-25 % for **both** the EMA teacher and
        Gemini-LLM estimates.
        """
        device = device or next(self.parameters()).device
        loader = self._get_train_dataloader(self.train_dataset)  # no shuffle

        mi_teacher, mi_llm, keep_filenames = [], [], []

        # ---------- 1-A. evaluate teacher MI (MC-dropout) ----------
        self.teacher_projector.eval()
        self.teacher_model.eval()
        upstream_model.eval()
        featurizer_model.eval()
        
        for wavs, _, filenames in loader:
            wavs = [w.to(device) for w in wavs]

            # upstream + featurizer
            h = upstream_model(wavs)
            feats = featurizer_model(wavs, h)
            feats = pad_sequence(feats, batch_first=True).to(device)
            feats = self.teacher_projector(feats)

            # MC-dropout passes
            outs = [torch.softmax(self.teacher_model(feats, None, mc=True)[0], -1) for _ in range(T)]                                       # (T,B,C)
            probs = torch.stack(outs, 0)                                     # (T,B,C)
            mean_p = probs.mean(0)                                           # (B,C)
            # entropy( mean_p )
            ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum(-1)      # (B,)
            # mean entropy
            ent_each = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean(0)
            mi = (ent_mean - ent_each).cpu()                                 # (B,)
            mi_teacher.extend(mi.tolist())

            keep_filenames.extend(filenames)                                 # align index

        # ---------- 1-B. evaluate LLM MI ----------------------------
        for fn in keep_filenames:
            if fn in self.gemini_multi_pred_dict:            # K-gen available
                multi = self.gemini_multi_pred_dict[fn].float().to(device)   # (K,C)
            else:                                            # fall back to 1-shot
                print(f"{fn}: fail to find multi-generation!")
                multi = self.gemini_dict[fn].unsqueeze(0).float().to(device) # (1,C)

            mean_p = multi.mean(0)
            ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum()
            ent_each = -(multi * multi.clamp_min(1e-8).log()).sum(-1).mean()
            mi_llm.append((ent_mean - ent_each).item())

        mi_teacher = np.array(mi_teacher)
        mi_llm     = np.array(mi_llm)
        

        plt.figure(figsize=(6,4), dpi=160)

        # keep both datasets on identical bin-edges → fair visual comparison
        common_bins = 60   # or use np.linspace(0, max(mi_teacher.max(), mi_llm.max()), 60)

        sns.histplot(mi_teacher,
                    bins=common_bins,
                    kde=True, stat="density",
                    color="tab:blue",  alpha=.35,
                    label="teacher")

        sns.histplot(mi_llm,
                    bins=common_bins,
                    kde=True, stat="density",
                    color="tab:orange", alpha=.35,
                    label="LLM")

        plt.title("Raw MI Teacher vs LLM distributions")
        plt.xlabel("Mutual Information")
        plt.ylabel("Density")
        plt.legend()

        out_path = os.path.join(self.expdir,
                                f"mi_distribution_epoch_{self.epoch:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        # sns.histplot(mi_teacher,  kde=True, color='tab:blue', label='teacher', stat='density')
        # sns.histplot(mi_llm,  kde=True, color='tab:orange', label='LLM',    stat='density')
        # plt.legend(); plt.title("Raw MI Tea vs LLM distributions")
        # plt.savefig(os.path.join(self.expdir, f'mi_distribution_epoch_{self.epoch:03d}.png'), dpi=200)
        # plt.close()

        # ---------- 1-C. joint-top-25 % filter ----------------------
        thr_t = np.quantile(mi_teacher, args.filter_thre) #0.90) #0.75) #
        thr_g = np.quantile(mi_llm,     args.filter_thre) #0.90) #0.75) #
        keep_indices = [i for i, (t, g) in enumerate(zip(mi_teacher, mi_llm))
                        if not (t > thr_t and g > thr_g)]

        featurizer_model.train()
        # upstream_model.train()
        self.epoch += 1
        return torch.utils.data.Subset(self.train_dataset, keep_indices)

    @torch.no_grad()
    def ent_mean_based_sample_selection(self,
                                featurizer_model,
                                upstream_model,
                                args,
                                device=None,
                                T: int = 8) -> torch.utils.data.Dataset:
        """
        Returns *filtered* torch.utils.data.Subset that excludes samples whose
        MI is simultaneously in the top-25 % for **both** the EMA teacher and
        Gemini-LLM estimates.
        """
        device = device or next(self.parameters()).device
        loader = self._get_train_dataloader(self.train_dataset)  # no shuffle

        ent_teacher, ent_llm, keep_filenames = [], [], []

        # ---------- 1-A. evaluate teacher MI (MC-dropout) ----------
        self.teacher_projector.eval()
        self.teacher_model.eval()
        upstream_model.eval()
        featurizer_model.eval()
        
        for wavs, _, filenames in loader:
            wavs = [w.to(device) for w in wavs]

            # upstream + featurizer
            h = upstream_model(wavs)
            feats = featurizer_model(wavs, h)
            feats = pad_sequence(feats, batch_first=True).to(device)
            feats = self.teacher_projector(feats)

            # MC-dropout passes
            outs = [torch.softmax(self.teacher_model(feats, None, mc=True)[0], -1) for _ in range(T)]                                       # (T,B,C)
            probs = torch.stack(outs, 0)                                     # (T,B,C)
            mean_p = probs.mean(0)                                           # (B,C)
            # entropy( mean_p )
            ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum(-1)      # (B,)
            # mean entropy
            # ent_each = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean(0)
            # mi = (ent_mean - ent_each).cpu()                                 # (B,)
            ent_teacher.extend(ent_mean.tolist())

            keep_filenames.extend(filenames)                                 # align index

        # ---------- 1-B. evaluate LLM MI ----------------------------
        for fn in keep_filenames:
            if fn in self.gemini_multi_pred_dict:            # K-gen available
                multi = self.gemini_multi_pred_dict[fn].float().to(device)   # (K,C)
            else:                                            # fall back to 1-shot
                print(f"{fn}: fail to find multi-generation!")
                multi = self.gemini_dict[fn].unsqueeze(0).float().to(device) # (1,C)

            mean_p = multi.mean(0)
            ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum()
            # ent_each = -(multi * multi.clamp_min(1e-8).log()).sum(-1).mean()
            ent_llm.append(ent_mean.item())

        ent_teacher = np.array(ent_teacher)
        ent_llm     = np.array(ent_llm)
        

        plt.figure(figsize=(6,4), dpi=160)

        # keep both datasets on identical bin-edges → fair visual comparison
        common_bins = 60   # or use np.linspace(0, max(mi_teacher.max(), mi_llm.max()), 60)

        sns.histplot(ent_teacher,
                    bins=common_bins,
                    kde=True, stat="density",
                    color="tab:blue",  alpha=.35,
                    label="teacher")

        sns.histplot(ent_llm,
                    bins=common_bins,
                    kde=True, stat="density",
                    color="tab:orange", alpha=.35,
                    label="LLM")

        plt.title("Raw Entropy Mean Teacher vs LLM distributions")
        plt.xlabel("Entropy Mean")
        plt.ylabel("Density")
        plt.legend()

        out_path = os.path.join(self.expdir,
                                f"ent_mean_distribution_epoch_{self.epoch:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        # sns.histplot(mi_teacher,  kde=True, color='tab:blue', label='teacher', stat='density')
        # sns.histplot(mi_llm,  kde=True, color='tab:orange', label='LLM',    stat='density')
        # plt.legend(); plt.title("Raw MI Tea vs LLM distributions")
        # plt.savefig(os.path.join(self.expdir, f'mi_distribution_epoch_{self.epoch:03d}.png'), dpi=200)
        # plt.close()

        # ---------- 1-C. joint-top-25 % filter ----------------------
        thr_t = np.quantile(ent_teacher, args.filter_thre) #0.90) #0.75) #
        thr_g = np.quantile(ent_llm,     args.filter_thre) #0.90) #0.75) #
        keep_indices = [i for i, (t, g) in enumerate(zip(ent_teacher, ent_llm))
                        if not (t > thr_t and g > thr_g)]

        featurizer_model.train()
        # upstream_model.train()
        self.epoch += 1
        return torch.utils.data.Subset(self.train_dataset, keep_indices)

    @torch.no_grad()
    def ent_each_based_sample_selection(self,
                                featurizer_model,
                                upstream_model,
                                args,
                                device=None,
                                T: int = 8) -> torch.utils.data.Dataset:
        """
        Returns *filtered* torch.utils.data.Subset that excludes samples whose
        MI is simultaneously in the top-25 % for **both** the EMA teacher and
        Gemini-LLM estimates.
        """
        device = device or next(self.parameters()).device
        loader = self._get_train_dataloader(self.train_dataset)  # no shuffle

        ent_teacher, ent_llm, keep_filenames = [], [], []

        # ---------- 1-A. evaluate teacher MI (MC-dropout) ----------
        self.teacher_projector.eval()
        self.teacher_model.eval()
        upstream_model.eval()
        featurizer_model.eval()
        
        for wavs, _, filenames in loader:
            wavs = [w.to(device) for w in wavs]

            # upstream + featurizer
            h = upstream_model(wavs)
            feats = featurizer_model(wavs, h)
            feats = pad_sequence(feats, batch_first=True).to(device)
            feats = self.teacher_projector(feats)

            # MC-dropout passes
            outs = [torch.softmax(self.teacher_model(feats, None, mc=True)[0], -1) for _ in range(T)]                                       # (T,B,C)
            probs = torch.stack(outs, 0)                                     # (T,B,C)
            # mean_p = probs.mean(0)                                           # (B,C)
            # entropy( mean_p )
            # ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum(-1)      # (B,)
            # mean entropy
            ent_each = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean(0)
            # mi = (ent_mean - ent_each).cpu()                                 # (B,)
            ent_teacher.extend(ent_each.tolist())

            keep_filenames.extend(filenames)                                 # align index

        # ---------- 1-B. evaluate LLM MI ----------------------------
        for fn in keep_filenames:
            if fn in self.gemini_multi_pred_dict:            # K-gen available
                multi = self.gemini_multi_pred_dict[fn].float().to(device)   # (K,C)
            else:                                            # fall back to 1-shot
                print(f"{fn}: fail to find multi-generation!")
                multi = self.gemini_dict[fn].unsqueeze(0).float().to(device) # (1,C)

            # mean_p = multi.mean(0)
            # ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum()
            ent_each = -(multi * multi.clamp_min(1e-8).log()).sum(-1).mean()
            ent_llm.append(ent_each.item())

        ent_teacher = np.array(ent_teacher)
        ent_llm     = np.array(ent_llm)
        

        plt.figure(figsize=(6,4), dpi=160)

        # keep both datasets on identical bin-edges → fair visual comparison
        common_bins = 60   # or use np.linspace(0, max(mi_teacher.max(), mi_llm.max()), 60)

        sns.histplot(ent_teacher,
                    bins=common_bins,
                    kde=True, stat="density",
                    color="tab:blue",  alpha=.35,
                    label="teacher")

        sns.histplot(ent_llm,
                    bins=common_bins,
                    kde=True, stat="density",
                    color="tab:orange", alpha=.35,
                    label="LLM")

        plt.title("Raw Entropy Each Teacher vs LLM distributions")
        plt.xlabel("Entropy Each")
        plt.ylabel("Density")
        plt.legend()

        out_path = os.path.join(self.expdir,
                                f"ent_each_distribution_epoch_{self.epoch:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        # sns.histplot(mi_teacher,  kde=True, color='tab:blue', label='teacher', stat='density')
        # sns.histplot(mi_llm,  kde=True, color='tab:orange', label='LLM',    stat='density')
        # plt.legend(); plt.title("Raw MI Tea vs LLM distributions")
        # plt.savefig(os.path.join(self.expdir, f'mi_distribution_epoch_{self.epoch:03d}.png'), dpi=200)
        # plt.close()

        # ---------- 1-C. joint-top-25 % filter ----------------------
        thr_t = np.quantile(ent_teacher, args.filter_thre) #0.90) #0.75) #
        thr_g = np.quantile(ent_llm,     args.filter_thre) #0.90) #0.75) #
        keep_indices = [i for i, (t, g) in enumerate(zip(ent_teacher, ent_llm))
                        if not (t > thr_t and g > thr_g)]

        featurizer_model.train()
        # upstream_model.train()
        self.epoch += 1
        return torch.utils.data.Subset(self.train_dataset, keep_indices)


    # def mi_based_sample_selection(self, featurizer_model, upstream_model):
        
    #     loader = self.get_train_dataloader()
    #     start_test = True
    #     # all_id = []
    #     # with torch.cuda.amp.autocast(enabled=amp): 
    #     upstream_model.eval()
    #     featurizer_model.eval()
    #     self.teacher_projector.eval()
    #     self.teacher_model.eval()
    #     with torch.no_grad():
    #         for batch_id, (wavs, labels, filenames) in enumerate(loader):
                
    #             with torch.cuda.amp.autocast(enabled=False):
    #                 wavs = [torch.FloatTensor(wav).to(args.device) for wav in wavs]
    #                 h_features = upstream_model(wavs)
    #                 feas = featurizer_model(wavs, h_features)
    #                 # print(f"after featurizer: {len(feas)}{feas[0].shape}")
    #                 features_len = torch.IntTensor([len(feat) for feat in feas]).to(device=args.device)
    #                 # labels: assumed to be soft labels with shape (batch_size, num_classes)
    #                 # features: [batch, feat_dim]
    #                 # feas = pad_sequence(feas, batch_first=True)
    #                 # print(f"after pad_sequence: {feas.shape}")
                    
    #                 with torch.no_grad():
    #                     feat_tea, _ = self.teacher_augmentation(feas)
    #                     feat_tea = pad_sequence(feat_tea, batch_first=True)
    #                     feat_tea = self.teacher_projector(feat_tea)
    #                     logits_det, _, _ = self.teacher_model(feat_tea, None, mc=False) 
    #                     teacher_probs = torch.softmax(logits_det, dim=-1)
    #                     # confidence, pseudo_labels = teacher_probs.max(dim=-1)
    #                     # mask = confidence >= self.confidence_threshold
    #                     # (b) MC-Dropout for reliability only
    #                     mean_teacher_probs, mi_teacher_probs = self.mc_teacher_uncert(feat_tea, T=8) # (B,)
    #                     #   turn MI into reliability weight
                        
    #                 feas = projector(feas)
    #                 # print(f"after projector: {feas.shape}")
    #                 logits, pooled, _ = classifier(feas, features_len)      # [batch, num_classes]
    #                 # print(f"logits: {len(logits[0])}")
    #             # print(f"filenames: {filenames}")
    #             if start_test:
    #                 all_fea = pooled.float().cpu()
    #                 all_output = logits.float().cpu()
    #                 all_label = labels.float().cpu()
    #                 all_id = filenames
    #                 all_teacher_mi = mi_teacher_probs.float().cpu()
    #                 # print(f"all_id: {all_id}")
    #                 start_test = False
    #             else:
    #                 all_fea = torch.cat((all_fea, pooled.float().cpu()), dim=0)
    #                 all_output = torch.cat((all_output, logits.float().cpu()), dim=0)
    #                 all_label = torch.cat((all_label, labels.float().cpu()), dim=0)
    #                 all_teacher_mi = torch.cat((all_teacher_mi, mi_teacher_probs.float().cpu), dim=0)
    #                 # print(f"all_id: {batch_id} {all_id}")
    #                 all_id.extend(filenames) #torch.cat((all_id, filenames), dim=0)
                

    #     # Apply softmax to outputs
    #     all_output = nn.Softmax(dim=1)(all_output)  # [N, C]
    #     print(f"all_output: {all_output.shape}")
    #     print(f"all_fea: {all_fea.shape}")
    #     print(f"all_label: {all_label.shape}")
        
    #     for fn in filenames:
    #         multi = self.gemini_multi_pred_dict[fn].to(device)
    #         mean_p = multi.mean(dim=0)
    #         ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum()
    #         ent_pass = -(multi * multi.clamp_min(1e-8).log()).sum(-1)
    #         mi       = ent_mean - ent_pass.mean()
    #         llm_mi_list.append(mi)
        
    #     featurizer_model.train()
        
    #     return dataset
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
    
    @torch.no_grad()
    def mc_teacher_uncert(self, proj_feat: torch.Tensor, T: int = 8):
        """
        proj_feat : (B,*,d) already passed through teacher_projector
        Returns
            mean_p  : (B,C)  ≈ deterministic softmax
            mi      : (B,)   mutual information (epistemic uncertainty)
        """
        self.teacher_model.eval()
        #logit, pooled, features_len
        outs = [torch.softmax(self.teacher_model(proj_feat, None, mc=True)[0], dim=-1) for _ in range(T)]                               # list[(B,C)]
        probs = torch.stack(outs, 0)    # (T,B,C)
        mean_p = probs.mean(0)          # (B,C)

        ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum(-1)      # (B,)
        ent_pass = -(probs * probs.clamp_min(1e-8).log()).sum(-1)        # (T,B)
        mi = ent_mean - ent_pass.mean(0)                                 # (B,)
        return mean_p, mi

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
    
    def compute_difficulty(self, prob_ema, prob_proxy):
        entropy = -(prob_ema * torch.log(prob_ema + 1e-6)).sum(dim=-1)
        agreement = F.cosine_similarity(prob_ema, prob_proxy, dim=-1)
        return entropy - agreement  # Lower = easier

    def dynamic_curriculum_mask(self, difficulty, pct):
        threshold = torch.quantile(difficulty, pct)
        return (difficulty <= threshold)  # Boolean mask

    def apply_acr_weights(self, sample_weights, epoch, total_epochs, eta=3.0):
        e_frac = epoch / total_epochs
        pinv = 1 - torch.exp(torch.tensor(-eta * e_frac))
        flip_mask = torch.rand_like(sample_weights) < pinv
        inverted_weights = torch.where(flip_mask, 1.0 - sample_weights, sample_weights)
        return inverted_weights
    def bidirectional_kl(self, p, q):
        """ KL(p‖q) + KL(q‖p)  (no /2 so we divide later if needed)"""
        kl_pq = (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(dim=-1)
        kl_qp = (q * (q.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(dim=-1)
        return 0.5 * (kl_pq + kl_qp)            # Eq. in fig. uses average


    def compute_r(self, prob_teacher, prob_gemini, alpha=5.0):
        """Exponential reliability score r = exp[ -α * (sym‑KL/2) ]"""
        sym_kl = self.bidirectional_kl(prob_teacher, prob_gemini)  # (B,)
        return torch.exp(-alpha * sym_kl)                      # (B,)


    def compute_w(self, r, epoch, total_epochs, beta=3.0, w_min=0.1, w_max=0.9):
        """ Final sample weight w = clamp( r * exp(-β·e′), w_min, w_max ) """
        e_frac = epoch / total_epochs
        pace = torch.exp(torch.tensor(-beta * e_frac, device=r.device))
        w = r * pace
        return w.clamp(min=w_min, max=w_max)

    # ------------------------------------------------------------
    # pooled : Tensor (N,D)    gt : Tensor (N,) with class-ids 0…C-1
    # ------------------------------------------------------------
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

        '''
        pooled = pooled.float()
        N, D   = pooled.shape
        assert len(gt) == N, "pooled and gt length mismatch"

        # ---------- 2-D PCA  ------------------------------------
        xy = PCA(n_components=2, whiten=True, random_state=0)\
            .fit_transform(pooled.numpy())                        # (N,2) 

        # ---------- class-coloured scatter (optional) ------------
        plt.figure(figsize=(4,4), dpi=160)
        scatter = plt.scatter(xy[:,0], xy[:,1],
                            c=gt.numpy(), cmap='tab10', s=8, alpha=.6)
        plt.xticks([]); plt.yticks([])
        plt.title('PCA-2D scatter'); plt.tight_layout()
        plt.savefig(Path(self.expdir) / f'{save_prefix}_scatter.png'); plt.close()

        # ---------- KDE on same xy ------------------------------
        kde   = gaussian_kde(xy.T, bw_method='scott')
        xmin, ymin = xy.min(0) - .05
        xmax, ymax = xy.max(0) + .05
        xx, yy   = np.mgrid[xmin:xmax:complex(n_grid),
                            ymin:ymax:complex(n_grid)]
        zz       = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        # ----- 2-D heat-map -------------------------------------
        plt.figure(figsize=(4,4), dpi=160)
        plt.imshow(zz.T, origin='lower',
                extent=[xmin,xmax,ymin,ymax], cmap=cmap)
        plt.xticks([]); plt.yticks([])
        plt.title('KDE heat-map'); plt.tight_layout()
        plt.savefig(Path(self.expdir) / f'{save_prefix}_kde2d.png'); plt.close()

        # ----- 3-D surface -------------------------------------
        fig = plt.figure(figsize=(5,4), dpi=160)
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, zz, cmap=cmap,
                        rcount=100, ccount=100,
                        linewidth=0, antialiased=False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((1,1,0.25))
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        plt.savefig(Path(self.expdir) / f'{save_prefix}_kde3d.png'); plt.close()
        '''

    # ------------------------------------------------------------
    # Example usage
    # pooled_dev = torch.load("pooled_dev_step5000.pt")   # (N,D)
    # gt_dev     = torch.load("labels_dev.pt")            # (N,)
    # plot_feature_kde(pooled_dev, gt_dev, save_prefix='step5000')



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
        
        features = pad_sequence(features, batch_first=True)
        #print(f"after pad_sequence: {features.shape}")
        self.teacher_projector.eval()
        self.teacher_model.eval()
        if mode == 'train':
            with torch.no_grad():
                if args.mc_sampling:
                    feat_tea, _ = self.teacher_augmentation(addi_features)
                    feat_tea = pad_sequence(feat_tea, batch_first=True)
                    feat_tea = self.teacher_projector(feat_tea)
                    logits_det, _, _ = self.teacher_model(feat_tea, None, mc=False) 
                    teacher_probs = torch.softmax(logits_det, dim=-1)
                    # confidence, pseudo_labels = teacher_probs.max(dim=-1)
                    # mask = confidence >= self.confidence_threshold
                    # (b) MC-Dropout for reliability only
                    mean_teacher_probs, mi_teacher_probs = self.mc_teacher_uncert(feat_tea, T=8) # (B,)
                    #   turn MI into reliability weight
                    entropy_teacher_probs = -(mean_teacher_probs * mean_teacher_probs.clamp_min(1e-8).log()).sum(-1) # mi_teacher_probs
                    if args.fusion_ent:
                        conf_teacher_probs = torch.exp(-3.0 * entropy_teacher_probs)
                    else:
                        conf_teacher_probs = torch.exp(-3.0 * mi_teacher_probs) #entropy_teacher_probs) # 
                else:
                    pseudo_labels, mask, teacher_probs = self.generate_pseudo_labels(addi_features) #features)
                

            gemini_probs = torch.stack([self.gemini_dict[fn] for fn in filenames], dim=0).to(device)
            
            features_aug, _ = self.student_augmentation(features)
            features_aug = pad_sequence(features_aug, batch_first=True)
            features_aug = self.student_projector(features_aug)
            if args.mc_sampling:
                student_logits, _, _ = self.student_model(features_aug, features_len, mc=False)
            else:
                student_logits, _, _ = self.student_model(features_aug, features_len)
            #print(f"logits shape: {student_logits.shape}")

            # teacher_soft_targets = teacher_probs  # shape [batch, #classes]

            # if global_step==1:
            #     print(f"pseudo label confidence mask: {mask.shape}")
            # selected_logits = student_logits[mask]
            # selected_teacher_probs = teacher_soft_targets[mask]
            
            if args.class_weights:
                class_weights = self.compute_class_weights_from_probs_thresholded(
                    teacher_probs,
                    method='effective-num',  # or 'effective-num', 'log-inverse'
                    threshold=1.0 / teacher_probs.shape[1]  # optional
                )
            else:
                class_weights = teacher_probs.new_ones(teacher_probs.size(-1))   # (C,)
                # print(f"shape of class_weights: {class_weights.shape}")

            # ----------------------------------------
            # Utility: Curriculum + ACR weight masking
            # ----------------------------------------
            # MatchOrConf Fusion (pseudo-label): Use Gemini and EMA teacher predictions
            mean_list, conf_list, uncert_list, mi_list, ent_list = [], [], [], [], [] #uncert_list = [], [], []
            if args.mc_sampling:
                # mean_list, conf_list, uncert_list = [], [], []
                alpha_llm = 3.0          # weighting hyper-param

                for fn in filenames:                                   # batch loop
                    if fn in self.gemini_multi_pred_dict:              # ---- K-shot available
                        multi = self.gemini_multi_pred_dict[fn].to(device)     # (K,C)
                        mean_p = multi.mean(dim=0)                              # (C,)

                        ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum()
                        ent_pass = -(multi * multi.clamp_min(1e-8).log()).sum(-1)  # (K,)
                        mi       = ent_mean - ent_pass.mean()                     # scalar
                        # uncert   = mi #ent_mean #                                             # use MI
                    else:                                             # ---- fall back to 1-shot
                        mean_p = self.gemini_dict[fn].to(device)                  # (C,)
                        ent_mean = -(mean_p * mean_p.clamp_min(1e-8).log()).sum()
                        mi = ent_mean
                        # uncert = mi #ent_mean  # entropy
                    if args.fusion_ent:
                        uncert = ent_mean
                    else:
                        uncert = mi

                    conf = torch.exp(-alpha_llm * uncert)                         # scalar

                    mean_list.append(mean_p)
                    conf_list.append(conf)
                    uncert_list.append(uncert)
                    mi_list.append(mi)
                    ent_list.append(ent_mean)

                # tensors for the whole batch
                mean_gemini_probs   = torch.stack(mean_list, 0)       # (B,C)
                conf_gemini_probs   = torch.stack(conf_list, 0)       # (B,)
                # entropy_gemini_probs = torch.stack(uncert_list, 0)    # (B,)
                entropy_gemini_probs = torch.stack(ent_list, 0)
                mi_gemini_probs = torch.stack(mi_list, 0)
                # gemini_multi_probs = torch.stack([self.gemini_multi_pred_dict[fn] if for fn in filenames], dim=0).to(device) # (B, K, C,)
                # mean_gemini_probs = gemini_multi_probs.mean(dim=1)            # (B , C)
                # ent_mean  = -(mean_gemini_probs * mean_gemini_probs.clamp_min(1e-8).log()).sum(-1)   # (B,) 
                # ent_pass  = -(gemini_multi_probs * gemini_multi_probs.clamp_min(1e-8).log()).sum(-1)  # (B,K)
                # mi_gemini    = ent_mean - ent_pass.mean(dim=1)
                # entropy_gemini_probs = mi_gemini
                # conf_gemini_probs = torch.exp(-3.0 * entropy_gemini_probs)
                
                entropy_gemini = -(mean_gemini_probs * mean_gemini_probs.log()).sum(dim=-1, keepdim=True)
                entropy_ema = -(mean_teacher_probs * mean_teacher_probs.log()).sum(dim=-1, keepdim=True)
                # ent_gap = abs(entropy_ema - entropy_gemini)
                if args.fusion_enabling:
                    if args.fusion_equal:
                        avg_weighted = 0.5 * (mean_gemini_probs + mean_teacher_probs)
                    else:
                        avg_weighted = (conf_teacher_probs.unsqueeze(1) * mean_teacher_probs + conf_gemini_probs.unsqueeze(1) * mean_gemini_probs ) / (conf_teacher_probs.unsqueeze(1) + conf_gemini_probs.unsqueeze(1)).clamp_min(1e-8) #(conf_teacher_probs.unsqueeze(1) * mean_teacher_probs + conf_gemini_probs.unsqueeze(1) * mean_gemini_probs ) / (conf_teacher_probs.unsqueeze(1) + conf_gemini_probs.unsqueeze(1)).clamp_min(1e-8) #(conf_teacher_probs.unsqueeze(1) * teacher_probs + conf_gemini_probs.unsqueeze(1) * gemini_probs ) / (conf_teacher_probs.unsqueeze(1) + conf_gemini_probs.unsqueeze(1)).clamp_min(1e-8)
                    if args.fu_thre == "KL":
                        kl_pq = (mean_teacher_probs * (mean_teacher_probs.clamp_min(1e-8).log() - mean_gemini_probs.clamp_min(1e-8).log())).sum(dim=-1, keepdim=True) #(mean_teacher_probs * (mean_teacher_probs.clamp_min(1e-8).log() - mean_gemini_probs.clamp_min(1e-8).log())).sum(dim=-1, keepdim=True)
                        kl_qp = (mean_gemini_probs * (mean_gemini_probs.clamp_min(1e-8).log() - mean_teacher_probs.clamp_min(1e-8).log())).sum(dim=-1, keepdim=True) #(mean_gemini_probs * (mean_gemini_probs.clamp_min(1e-8).log() - mean_teacher_probs.clamp_min(1e-8).log())).sum(dim=-1, keepdim=True)
                        sym_kl = 0.5 * (kl_pq + kl_qp)
                        
                        # tau     = args.cos_tau #0.9
                        # cos_sim = F.cosine_similarity(mean_gemini_probs, mean_teacher_probs, dim=-1)  #F.cosine_similarity(gemini_probs, teacher_probs, dim=-1) # (B,)
                        # cos_sim = cos_sim.unsqueeze(1)
                        pseudo_label = torch.where(
                            sym_kl < args.kl_tau, #0.1, #0.06 #cos_sim > tau, #ent_gap < 0.08, #
                            avg_weighted,
                            torch.where(entropy_gemini < entropy_ema, mean_gemini_probs, mean_teacher_probs) #mean_gemini_probs, mean_teacher_probs) #gemini_probs, teacher_probs)
                        )
                        src_tag = torch.full((pseudo_label.size(0),), 2, dtype=torch.int8, device=pseudo_label.device)
                        src_tag[sym_kl.squeeze(1) < args.kl_tau] = 0
                        src_tag[(sym_kl.squeeze(1)>= args.kl_tau) & (entropy_gemini.squeeze(1) < entropy_ema.squeeze(1))] = 1
                    
                    elif args.fu_thre == "N/A":
                        pseudo_label = avg_weighted
                        src_tag = torch.full((pseudo_label.size(0),), 0, dtype=torch.int8, device=pseudo_label.device)
                
                else:
                    pseudo_label = torch.where(entropy_gemini < entropy_ema, mean_gemini_probs, mean_teacher_probs)
                    src_tag = torch.full((pseudo_label.size(0),), 2, dtype=torch.int8, device=pseudo_label.device)
                    src_tag[entropy_gemini.squeeze(1) < entropy_ema.squeeze(1)] = 1
                
                
                # src_tag[ent_gap.squeeze(1) < 0.08] = 0
                # src_tag[(ent_gap.squeeze(1)>= 0.08) & (entropy_gemini.squeeze(1) < entropy_ema.squeeze(1))] = 1
                # src_tag[cos_sim.squeeze(1) > tau] = 0
                # src_tag[(cos_sim.squeeze(1) <= tau) & (entropy_gemini.squeeze(1) < entropy_ema.squeeze(1))] = 1
            else:
                entropy_gemini = -(gemini_probs * gemini_probs.log()).sum(dim=-1, keepdim=True)
                entropy_ema = -(teacher_probs * teacher_probs.log()).sum(dim=-1, keepdim=True)
                # print(f"gemini: {gemini_probs.shape}; teacher: {teacher_probs.shape}")
                # print(f"entropy gemini: {entropy_gemini.shape}, teacher: {entropy_ema.shape}")
                if args.fusion_enabling:
                    if args.fusion_equal:
                        avg_weighted = 0.5 * (gemini_probs + teacher_probs)
                    else:
                        alpha_llm = 3.0
                        alpha_tea = 3.0
                        entropy_gemini_probs = -(gemini_probs * gemini_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
                        entropy_teacher_probs = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
                        conf_gemini_probs = torch.exp(-alpha_llm * entropy_gemini_probs)
                        conf_teacher_probs = torch.exp(-alpha_tea * entropy_teacher_probs)
                        avg_weighted = (conf_teacher_probs * teacher_probs + conf_gemini_probs * gemini_probs ) / (conf_teacher_probs + conf_gemini_probs).clamp_min(1e-8)
                    
                    if args.fu_thre == "KL":
                        # tau     = args.sim_tau
                        # cos_sim = F.cosine_similarity(gemini_probs, teacher_probs, dim=-1) # Why not use bidirectional KL divergence to calculate similarity?
                        # cos_sim = cos_sim.unsqueeze(1) 
                        # print(f"cos_sim: {cos_sim.shape}")
                        kl_pq = (teacher_probs * (teacher_probs.clamp_min(1e-8).log() - gemini_probs.clamp_min(1e-8).log())).sum(dim=-1, keepdim=True) #(mean_teacher_probs * (mean_teacher_probs.clamp_min(1e-8).log() - mean_gemini_probs.clamp_min(1e-8).log())).sum(dim=-1, keepdim=True)
                        kl_qp = (gemini_probs * (gemini_probs.clamp_min(1e-8).log() - teacher_probs.clamp_min(1e-8).log())).sum(dim=-1, keepdim=True) #(mean_gemini_probs * (mean_gemini_probs.clamp_min(1e-8).log() - mean_teacher_probs.clamp_min(1e-8).log())).sum(dim=-1, keepdim=True)
                        sym_kl = 0.5 * (kl_pq + kl_qp)
                        
                        pseudo_label = torch.where(
                            sym_kl < args.kl_tau, #tau, #0.9,
                            avg_weighted,
                            torch.where(entropy_gemini < entropy_ema, gemini_probs, teacher_probs)
                        ) # the fusion weights between gemini_probs and teacher_probs could be analyzed further, e.g. reliability score(entropy/confidence) from MC dropout and average of multi-generation
                    
                        src_tag = torch.full((pseudo_label.size(0),), 2, dtype=torch.int8, device=pseudo_label.device)
                        src_tag[(sym_kl.squeeze(1) < args.kl_tau)] = 0
                        src_tag[(sym_kl.squeeze(1) >= args.kl_tau) & (entropy_gemini.squeeze(1) < entropy_ema.squeeze(1))] = 1
                        # src_tag[(cos_sim.squeeze(1) > 0.9)] = 0                                  # fused
                        # src_tag[(cos_sim.squeeze(1) <= 0.9) & (entropy_gemini.squeeze(1) < entropy_ema.squeeze(1))] = 1  # gemini wins
                    elif args.fu_thre == "N/A":
                        pseudo_label = avg_weighted
                        src_tag = torch.full((pseudo_label.size(0),), 0, dtype=torch.int8, device=pseudo_label.device)
                else:
                    pseudo_label = torch.where(entropy_gemini < entropy_ema, gemini_probs, teacher_probs)
                    src_tag = torch.full((pseudo_label.size(0),), 2, dtype=torch.int8, device=pseudo_label.device)
                    src_tag[entropy_gemini.squeeze(1) < entropy_ema.squeeze(1)] = 1
                    
                        
            # ---- LOG to file ---------------------------------------------------
            log_path = Path(self.expdir) / "pseudo_fusion_log.tsv"
            with open(log_path, "a") as lf:
                for fn, tag in zip(filenames, src_tag.cpu().tolist()):
                    lf.write(f"{global_step}\t{fn}\t{tag}")
            
            pseudo_class = torch.argmax(pseudo_label, dim=1)
            labels = labels.to(features.device)
            true_class = torch.argmax(labels, dim=1)

            # For classification report (CPU numpy arrays)
            pseudo_np = pseudo_class.cpu().numpy(force=True)
            labels_np = true_class.cpu().numpy(force=True)

            predictions_binary = torch.zeros_like(pseudo_label, dtype=torch.float)  # (B, C)
            predictions_binary.scatter_(1, pseudo_class.unsqueeze(1), 1.0)

            labels_binary = torch.zeros_like(labels, dtype=torch.float)          # (B, C)
            labels_binary.scatter_(1, true_class.unsqueeze(1), 1.0)
            # prediction_distribution = torch.from_numpy(aff)
            # predictions_binary = torch.where(prediction_distribution > self.k_thresold, 1.0, 0.0)
            # labels_binary = torch.where(all_label > self.k_thresold, 1.0, 0.0)

            all_emotions = self.all_emotions
            reprot_dict = classification_report(
                labels_binary.cpu().numpy(force=True),
                predictions_binary.cpu().numpy(force=True),
                target_names=all_emotions,
                output_dict=True
            )
            macro_f1 = reprot_dict['macro avg']['f1-score']
            acc = accuracy_score(labels_np, pseudo_np)
            precision = reprot_dict['macro avg']['precision']
            recall = reprot_dict['macro avg']['recall']
            
            log_str = f'step {global_step}, acc = {acc * 100: .2f}%\nprecision = {precision*100:.2f}%\nrecall = {recall*100:.2f}%\nmacro f1 = {macro_f1 * 100:.2f}%\n'
            
            output_file = open(Path(self.expdir) / "pseudo_label_performance_records.txt", "a")
            output_file.write(log_str + '\n')
            output_file.flush()
            # print(log_str + '\n')
            
            

            # print(f"fused pseudo label: {pseudo_label.shape}")
            # with open(Path(self.expdir) / "log.log", 'a') as f:
            # log_str = f'acc = {acc * 100: .2f}%\nprecision = {precision*100:.2f}%\nrecall = {recall*100:.2f}%\nmacro f1 = {macro_f1 * 100:.2f}% (soft target match)\nhamming accuracy = {hamming_acc * 100:.2f}'
            
            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            # print(log_str + '\n')
            # Step 2: Curriculum Masking
            with torch.no_grad():
                if args.mc_sampling:
                    difficulty = self.compute_difficulty(mean_teacher_probs, mean_gemini_probs) #mean_teacher_probs, mean_gemini_probs) # teacher_probs, gemini_probs)
                else:
                    difficulty = self.compute_difficulty(teacher_probs, gemini_probs)
            if args.use_curriculum:
                with torch.no_grad():
                    easy_mask = self.dynamic_curriculum_mask(difficulty, pct=0.8)  # Example: top 80% easiest samples
            else:
                with torch.no_grad():
                    easy_mask = torch.ones_like(difficulty, dtype=torch.bool)

            if args.use_acr:
                if args.mc_sampling:
                    r_score = self.compute_r(mean_teacher_probs, mean_gemini_probs, alpha=5.0) #mean_teacher_probs, mean_gemini_probs, alpha=5.0)  # tune α # teacher_probs, gemini_probs, alpha=5.0)
                else:
                    r_score = self.compute_r(teacher_probs, gemini_probs, alpha=5.0) 
                w_sample = self.compute_w(r_score, epoch=global_step, total_epochs=self.total_epochs, beta=3.0, w_min=0.1, w_max=0.9) #int(global_step*32/5512), total_epochs=self.total_epochs, beta=3.0, w_min=0.1, w_max=0.9)
                sample_weights = w_sample * easy_mask.float()
            else:
                sample_weights = easy_mask.float()   # 1 for kept, 0 for filtered
            
            log_path = Path(self.expdir) / "pseudo_diagnostics.tsv"
            header   = "\t".join(
                ["step","file","gt","src","correct",
                "sym_kl","e_gem","e_ema","ent_gem","ent_tea","mi_gem","mi_tea","conf_gem","conf_tea","w_used"]) + "\n" #"ent_gem","ent_tea","conf_gem","conf_tea","w_used"]) + "\n"  #"cos""ent_gap"

            if global_step == 1 and not log_path.exists():
                log_path.write_text(header)            # create header once

            with log_path.open("a") as lf:
                for i, fn in enumerate(filenames):
                    correct = int(pseudo_class[i] == true_class[i])
                    if args.mc_sampling:
                        row = [
                            f"{global_step}",
                            fn,
                            f"{true_class[i]}",
                            f"{src_tag[i].item()}",
                            f"{correct}",
                            f"{sym_kl[i,0].item():.4f}" if args.fusion_enabling and args.fu_thre == "KL" else "NA", #f"{cos_sim[i,0].item():.4f}", #f"{ent_gap[i,0].item():.4f}", #
                            f"{entropy_gemini[i,0].item():.4f}",
                            f"{entropy_ema[i,0].item():.4f}",
                            f"{entropy_gemini_probs[i].item():.4f}" if args.mc_sampling else "NA",
                            f"{entropy_teacher_probs[i].item():.4f}" if args.mc_sampling else "NA",
                            f"{mi_gemini_probs[i].item():.4f}" if args.mc_sampling else "NA",
                            f"{mi_teacher_probs[i].item():.4f}" if args.mc_sampling else "NA", #f"{entropy_teacher_probs[i].item():.4f}" if args.mc_sampling else "NA", #
                            f"{conf_gemini_probs[i].item():.4f}" if args.mc_sampling else "NA",
                            f"{conf_teacher_probs[i].item():.4f}" if args.mc_sampling else "NA",
                            f"{sample_weights[i].item():.4f}"
                        ]
                    else:
                        row = [
                            f"{global_step}",
                            fn,
                            f"{true_class[i]}",
                            f"{src_tag[i].item()}",
                            f"{correct}",
                            f"{sym_kl[i,0].item():.4f}" if args.fusion_enabling and args.fu_thre == "KL" else "NA", #f"{cos_sim[i,0].item():.4f}", #f"{ent_gap[i,0].item():.4f}", #
                            f"{entropy_gemini[i,0].item():.4f}",
                            f"{entropy_ema[i,0].item():.4f}",
                            f"{entropy_gemini_probs[i].item():.4f}" if args.fusion_enabling and not args.fusion_equal else "NA",
                            f"{entropy_teacher_probs[i].item():.4f}" if args.fusion_enabling and not args.fusion_equal else "NA",
                            f"{conf_gemini_probs[i].item():.4f}" if args.fusion_enabling and not args.fusion_equal else "NA",
                            f"{conf_teacher_probs[i].item():.4f}" if args.fusion_enabling and not args.fusion_equal else "NA",
                            f"{sample_weights[i].item():.4f}"
                        ]
                    lf.write("\t".join(row) + "\n")
            # Step 3: ACR Reweighting
            # 3) Reliability r and pace‑weighted w (ACR‑style)
            # r_score = self.compute_r(teacher_probs, gemini_probs, alpha=5.0)  # tune α
            # w_sample = self.compute_w(r_score, epoch=global_step, total_epochs=self.total_epochs,
            #                     beta=3.0, w_min=0.1, w_max=0.9)
            # apply curriculum mask
            # sample_weights = w_sample * easy_mask.float()

            # 4) Distillation loss (KL) with sample_weights
            # print(f"student_logits: {student_logits.shape}")
            logits_sel = student_logits
            targets_sel = pseudo_label
            # loss_per = F.kl_div(F.log_softmax(logits_sel, dim=-1), targets_sel, reduction='none').sum(dim=1)
            # weighted_loss = (sample_weights * loss_per).mean()

            # raw_weights = easy_mask.float()
            # weights = self.apply_acr_weights(raw_weights, epoch=global_step, total_epochs=self.total_epochs)

            # Step 4: Masked distillation loss
            # mask = 
            # selected_logits = student_logits[mask]
            # selected_targets = pseudo_label[mask]
            # loss_vec = F.kl_div(F.log_softmax(selected_logits, dim=-1), selected_targets, reduction='none').sum(dim=1)
            # distill_loss = (weights[mask] * loss_vec).mean()

            
            # EMA distillation loss: Consistency Loss + Pseudo Labeling 
            # if selected_logits.shape[0] == 0:
            #     distill_loss = torch.tensor(0.0, device=device, requires_grad=True)
            # else:
            loss_per = self.objective(
                logits_sel,
                targets_sel,
                class_weights.to(device), # self.class_balanced_weights.to(device), # 
                reduction='none' #'mean'
            )
            distill_loss = (sample_weights * loss_per).sum() / sample_weights.sum().clamp_min(1) #(sample_weights * loss_per).mean()
            # print(f"distill_loss: {distill_loss}")
        
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
                # print(f"ent_loss: {ent_loss}")
            else:
                ent_loss = torch.tensor(0.0, device=device, requires_grad=True)

            lambda_ent = args.lambda_ent # 0.1
            
            # -------------------------------
            # diversity loss
            # -------------------------------
            student_probs = torch.softmax(student_logits, dim=-1)   
            mean_probs = student_probs.mean(dim=0)
            div_loss = (class_weights * mean_probs * torch.log(mean_probs + 1e-6)).sum()
            # print(f"div_loss: {div_loss}")
            lambda_div = args.lambda_div
            
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
                # print(f"fbnm_loss: {fbnm_loss}")
            else:
                fbnm_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
            lambda_fbnm = args.lambda_fbnm # 0.1
            
            final_loss = distill_loss + lambda_div * div_loss + lambda_ent * ent_loss + lambda_fbnm * fbnm_loss
            
            self._ema_update_teacher()
            
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
            if args.mc_sampling:
                predicted, pooled, _ = self.student_model(features, features_len, mc=False)
            else:
                predicted, pooled, _ = self.student_model(features, features_len)
            if global_step in {1, int(self.total_steps*0.05), int(self.total_steps*0.5), self.total_steps} and mode=="dev":
                gt_dev = torch.argmax(labels.to(features.device), dim=1)
                # print(f"dev step {global_step} pooled: {pooled.shape}, gt_dev: {gt_dev.shape}") 
                vis_feats.append(pooled.cpu())
                vis_labs.append(gt_dev.cpu())
            # if global_step in {1, int(self.total_steps*0.05), int(self.total_steps*0.5), self.total_steps} and mode == 'dev': # global_step % 500 == 0 and mode == 'dev':
            #     np.save(f"{self.expdir}/feats_step{global_step}.npy", pooled.cpu().numpy())
            #     gt_dev = torch.argmax(labels.to(features.device), dim=1) 
            #     self.plot_feature_kde(pooled, gt_dev, save_prefix=f'step{global_step}')
                
                '''
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, whiten=True)   # (N,2)
                xy   = pca.fit_transform(pooled.numpy()) 
                from scipy.stats import gaussian_kde
                import numpy as np

                kde = gaussian_kde(xy.T, bw_method='scott')   # Scott’s rule

                # grid limits (slightly larger than data range)
                xmin, ymin = xy.min(0) - 0.05
                xmax, ymax = xy.max(0) + 0.05
                xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]      # 200×200 grid
                coords = np.vstack([xx.ravel(), yy.ravel()])

                zz = kde(coords).reshape(xx.shape)     # density on the grid import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D       # noqa: F401

                fig = plt.figure(figsize=(3,3), dpi=150)
                ax  = fig.add_subplot(111, projection='3d')

                # ax.plot_surface uses X, Y, Z of equal shape
                surf = ax.plot_surface(xx, yy, zz,
                                    cmap='jet',  # red=high density
                                    rcount=100, ccount=100,
                                    antialiased=False,
                                    linewidth=0)

                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.set_box_aspect((1,1,0.25))           # flatten look
                ax.view_init(elev=30, azim=45)          # similar viewpoint
                plt.tight_layout()
                plt.show()
                plt.imshow(zz.T, origin='lower', extent=[xmin,xmax,ymin,ymax], cmap='jet')
                plt.axis('off')
                plt.show()
                '''
            labels = labels.to(features.device)
            
            with torch.no_grad():
                # For unlabeled or target domain adaptation scenario:
                # If you want to combine real labels + pseudo labels, adapt this logic.
                if args.mc_sampling:
                    feat_tea, _ = self.teacher_augmentation(addi_features)
                    feat_tea = pad_sequence(feat_tea, batch_first=True)
                    feat_tea = self.teacher_projector(feat_tea)
                    logits_det, _, _ = self.teacher_model(feat_tea, None, mc=False) 
                    teacher_probs = torch.softmax(logits_det, dim=-1)
                    confidence, pseudo_labels = teacher_probs.max(dim=-1)
                    mask = confidence >= self.confidence_threshold
                    
                else:
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
            # prediction_distribution = torch.nn.functional.softmax(predicted, dim=1)
            predictions_binary = torch.zeros_like(predicted, dtype=torch.float)  # (B, C)
            predictions_binary.scatter_(1, predicted_class.unsqueeze(1), 1.0)

            labels_binary = torch.zeros_like(labels, dtype=torch.float)          # (B, C)
            labels_binary.scatter_(1, true_class.unsqueeze(1), 1.0)
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

'''

'''