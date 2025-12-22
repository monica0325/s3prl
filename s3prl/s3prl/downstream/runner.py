import os
import sys
import math
import glob
import uuid
import shutil
import random
import tempfile
import importlib
from pathlib import Path
import copy

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.distributed import is_initialized, get_rank, get_world_size
import torch.nn.functional as F
import torch.nn as nn

from s3prl import hub
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict

from huggingface_hub import HfApi, HfFolder, Repository

SAMPLE_RATE = 16000

MODEL_CARD_MARKDOWN = """---
datasets:
- superb
tags:
- library:s3prl
- benchmark:superb
- type:model
---

# Fine-tuned s3prl model

Upstream Model: {upstream_model}

## Model description

[More information needed]

## Intended uses & limitations

[More information needed]

## How to use

[More information needed]

## Limitations and bias

[More information needed]

## Training data

[More information needed]

## Training procedure

[More information needed]

## Evaluation results

[More information needed]

"""


class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces

def ema_update(stu, tea, tau=0.999):
    with torch.no_grad():
        for ps, pt in zip(stu.parameters(), tea.parameters()):
            pt.data.mul_(tau).add_(ps.data, alpha=1 - tau)

class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        self.source_ckpt = (
            torch.load(args.source_ckpt, map_location="cpu")
            if getattr(args, "source_ckpt", None)
            else None
        )
        # self.adapt_feat = getattr(args, "adapt_feature_extractor", False)

        self.upstream = self._get_upstream()
        self.featurizer = self._get_featurizer()
        if self.config.get('num_featurizer', None):
            self.additional_featurizers = self._get_additional_featurizer()
        else:
            self.additional_featurizers = None
        self.bank_store = args.bank_store
        
        self.downstream = self._get_downstream()
        self.all_entries = [self.upstream, self.featurizer, self.downstream]
        if self.additional_featurizers:
            self.all_entries += [self.additional_featurizers]
        
        
        if self.args.upstream_trainable and "distill" in self.args.downstream:
            from .noise import AddNoise 
            from .specaug import SpecAug
            self.tea_aug   = AddNoise(noise_mean=0.0, noise_std=0.005, intensity=1.0) # weak: AddNoise(noise_std=0.002)
            self.stu_aug = SpecAug( # strong
                apply_time_warp=True,
                time_warp_window=5,
                apply_freq_mask=True,
                freq_mask_width_range=(0, 20),
                apply_time_mask=True,
                time_mask_width_range=(0, 100),
            )
            self.additional_upstream = self._get_additional_upstream()
            self.all_entries += [self.additional_upstream]
        else:
            self.additional_upstream = None
            self.tea_aug = None
            self.stu_aug = None
        
    def _bank_initialization(self, ):
        return    
    
    def _duplicate_for_student_teacher(self, raw_state: dict, stu_prefix="student", tea_prefix="teacher"):
        """
        Map     model.xxx → student_model.xxx  &  teacher_model.xxx
        Keeps   other keys unchanged.
        """
        out = {}
        for k, v in raw_state.items():
            if k.startswith("model."):
                suffix = k[len("model."):]
                out[f"{stu_prefix}_model.{suffix}"] = v.clone()
                out[f"{tea_prefix}_model.{suffix}"] = v.clone()
            elif k.startswith("projector."):
                suffix = k[len("projector."):]
                out[f"{stu_prefix}_projector.{suffix}"] = v.clone()
                out[f"{tea_prefix}_projector.{suffix}"] = v.clone()
            else:
                out[k] = v
        return out

        
    def _load_weight(self, model, name, prefer_source=False):
        state = None
        if prefer_source and self.source_ckpt:
            state = self.source_ckpt.get(name)

        if state is None:
            state = self.init_ckpt.get(name)

        # if state:
        #     show(f"[Runner] ‑ Loading weights for {name}")
        #     model.load_state_dict(state, strict=True)
        if state:
            show(f'[Runner] - Loading weights for {name}')
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as e:
                # Detect the specific student / teacher mismatch
                if ("student_model" in str(e) or "teacher_model" in str(e) or "student_projector" in str(e) or "teacher_projector") and \
                hasattr(model, "student_model") and hasattr(model, "teacher_model") and hasattr(model, "student_projector") and hasattr(model, "teacher_projector"):
                    fixed_state = self._duplicate_for_student_teacher(state)
                    model.load_state_dict(fixed_state, strict=False) #True
                    show('[Runner] - State-dict adapted for student/teacher twin heads')
                else:
                    raise
        # init_weight = self.init_ckpt.get(name)
        # if init_weight:
        #     show(f'[Runner] - Loading {name} weights from the previous experiment')
        #     model.load_state_dict(init_weight)


    def _init_model(self, model, name, trainable, interfaces=None, prefer_source=False):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name, prefer_source)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)


    def _get_upstream(self):
        if "from_hf_hub" in self.args and self.args.from_hf_hub == True:
            from huggingface_hub import snapshot_download

            print(f'[Runner] - Downloading upstream model {self.args.upstream} from the Hugging Face Hub')
            filepath = snapshot_download(self.args.upstream, self.args.upstream_revision, use_auth_token=True)
            sys.path.append(filepath)

            dependencies = (Path(filepath) / 'requirements.txt').resolve()
            print("[Dependency] - The downloaded upstream model requires the following dependencies. Please make sure they are installed:")
            for idx, line in enumerate((Path(filepath) / "requirements.txt").open().readlines()):
                print(f"{idx}. {line.strip()}")
            print(f"You can install them by:")
            print()
            print(f"pip install -r {dependencies}")
            print()

            from expert import UpstreamExpert
            Upstream = UpstreamExpert
            ckpt_path = os.path.join(filepath, self.args.upstream_model_name)
        else:
            Upstream = getattr(hub, self.args.upstream)
            ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)
        
        for p in model.model.feature_extractor.parameters():
            p.requires_grad = False
        # (Optionally) freeze the conv LayerNorm & post-projection too:
        for p in model.model.layer_norm.parameters():
            p.requires_grad = False
        if getattr(model, "post_extract_proj", None) is not None:
            for p in model.model.post_extract_proj.parameters():
                p.requires_grad = False      # 512→768 linear

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model = model,
            name = 'Upstream',
            trainable = self.args.upstream_trainable,
            interfaces = ["get_downsample_rates"]
        )
    def _get_additional_upstream(self):
        if "from_hf_hub" in self.args and self.args.from_hf_hub == True:
            from huggingface_hub import snapshot_download

            print(f'[Runner] - Downloading upstream model {self.args.upstream} from the Hugging Face Hub')
            filepath = snapshot_download(self.args.upstream, self.args.upstream_revision, use_auth_token=True)
            sys.path.append(filepath)

            dependencies = (Path(filepath) / 'requirements.txt').resolve()
            print("[Dependency] - The downloaded upstream model requires the following dependencies. Please make sure they are installed:")
            for idx, line in enumerate((Path(filepath) / "requirements.txt").open().readlines()):
                print(f"{idx}. {line.strip()}")
            print(f"You can install them by:")
            print()
            print(f"pip install -r {dependencies}")
            print()

            from expert import UpstreamExpert
            Upstream = UpstreamExpert
            ckpt_path = os.path.join(filepath, self.args.upstream_model_name)
        else:
            Upstream = getattr(hub, self.args.upstream)
            ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)
        
        for p in model.model.parameters():
            p.requires_grad = False
        
        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model = model,
            name = 'Upstream',
            trainable = False,
            interfaces = ["get_downsample_rates"]
        )


    def _get_featurizer(self):
        model = Featurizer(
            upstream = self.upstream.model,
            feature_selection = self.args.upstream_feature_selection,
            layer_selection = self.args.upstream_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = self.args.featurizer_trainable, #True,
            interfaces = ['output_dim', 'downsample_rate'],
            prefer_source=True,
        )


    def _get_additional_featurizer(self):
        model = Featurizer(
            upstream = self.upstream.model,
            feature_selection = self.args.upstream_feature_selection,
            layer_selection = self.args.upstream_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
        ).to(self.args.device)
        
        # for p in model.parameters():
        #     p.requires_grad = False   # EMA teacher stays frozen (no grads)

        return self._init_model(
            model = model,
            name = 'AdditionalFeaturizer',
            trainable = False,
            interfaces = ['output_dim', 'downsample_rate'],
            prefer_source=True,
        )
        
        
    def _get_downstream(self):
        expert = importlib.import_module(f"downstream.{self.args.downstream}.expert")
        Downstream = getattr(expert, "DownstreamExpert")

        model = Downstream(
            upstream_dim = self.featurizer.model.output_dim,
            upstream_rate = self.featurizer.model.downsample_rate,
            **self.config,
            **vars(self.args)
        ).to(self.args.device)

        if hasattr(model, "teacher_model"):
            for p in model.teacher_model.parameters():
                p.requires_grad = False   # EMA teacher stays frozen (no grads)
        if hasattr(model, "teacher_projector"):
            for p in model.teacher_projector.parameters():
                p.requires_grad = False   # EMA teacher stays frozen (no grads)
        
        if not self.args.downstream_trainable:
            for p in model.parameters(): 
                p.requires_grad = False

        return self._init_model(
            model = model,
            name = 'Downstream',
            trainable = self.args.downstream_trainable, #True,
            interfaces = ['get_dataloader', 'log_records'],
            prefer_source=True,
        )


    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )
        self._load_weight(optimizer, 'Optimizer')
        return optimizer


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )
        self._load_weight(scheduler, 'Scheduler')
        return scheduler

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(upstream_model=self.args.upstream)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)


    def train(self):
        # trainable parameters and train/eval mode
        trainable_models = []
        trainable_paras = []
        for entry in self.all_entries:
            if entry.trainable:
                entry.model.train().to(self.args.device)
                trainable_models.append(entry.model)
                trainable_paras += [p for p in entry.model.parameters() if p.requires_grad]
                # trainable_paras += list(entry.model.parameters())
            else:
                entry.model.eval()

        # set amp
        amp = self.config['runner'].get('fp16', False)
        if amp:
            print('[Runner] - Enabled fp16 training')
            scaler = torch.cuda.amp.GradScaler()

        # optimizer
        optimizer = self._get_optimizer(trainable_models)

        # scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)

        # specaug
        specaug = None
        # addnoise = None
        if self.config.get('specaug'):
            from .specaug import SpecAug
            specaug = SpecAug(**self.config["specaug"])
        # if self.upstream.trainable and "distill" in self.args.downstream:
        #     from .noise import AddNoise
        #     addnoise = AddNoise(noise_mean=0.0, noise_std=0.005, intensity=1.0)

        # progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        pbar = tqdm(total=self.config['runner']['total_steps'], dynamic_ncols=True, desc='overall', file=tqdm_file)
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)
            
        batch_ids = []
        backward_steps = 0
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', 0)
        train_split = self.config['runner'].get("train_dataloader", "train")
        
        # Feature/Score Bank Initialization
        if self.bank_store and self.upstream.trainable and "distill" in self.args.downstream:
            self.upstream.model.eval()
            self.featurizer.model.eval()
            if self.additional_upstream:
                self.additional_upstream.model.eval()
            if self.additional_featurizers:
                self.additional_featurizers.model.eval()
            loader = self.downstream.model.get_dataloader(train_split)
            num_sample=len(loader.dataset)
            num_classes = loader.dataset.lab_list.shape[-1]
            self.fea_bank=torch.randn(num_sample,256)
            self.score_bank = torch.randn(num_sample, num_classes).cuda()
            
            with torch.no_grad():
                for batch_id, (wavs, labels, filenames, indx) in enumerate(loader):
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                        h_features = self.upstream.model(wavs)
                        feas = self.featurizer.model(wavs, h_features)
                        # print(f"after featurizer: {len(feas)}{feas[0].shape}")
                        features_len = torch.IntTensor([len(feat) for feat in feas]).to(device=self.args.device)
                        # labels: assumed to be soft labels with shape (batch_size, num_classes)
                        # features: [batch, feat_dim]
                        feas = pad_sequence(feas, batch_first=True)
                        # print(f"after pad_sequence: {feas.shape}")
                        
                        feas = self.downstream.model.projector(feas)
                        # print(f"after projector: {feas.shape}")
                        logits, pooled, _ = self.downstream.model.model(feas, features_len)      # [batch, num_classes]
                        # print(f"logits: {len(logits[0])}")
                        output_norm = F.normalize(pooled)
                        outputs = nn.Softmax(-1)(logits)
                        self.fea_bank[indx] = output_norm.detach().clone().cpu()
                        self.score_bank[indx] = outputs.detach().clone() #.cpu()
            
            self.downstream.model.projector.train()
            self.downstream.model.model.train()
            
            self.upstream.model.train()
            self.featurizer.model.train()
        elif self.bank_store:
            # self.downstream.model.projector.eval()
            if "distillation" in self.args.downstream:
                self.downstream.model.student_projector.eval()
                self.downstream.model.student_model.eval()
                self.downstream.model.teacher_projector.eval()
                self.downstream.model.teacher_model.eval()
            else:
                self.downstream.model.projector.eval()
                self.downstream.model.model.eval()
            self.upstream.model.eval()
            self.featurizer.model.eval()
            if self.additional_featurizers:
                self.additional_featurizers.model.eval()
            loader = self.downstream.model.get_dataloader(train_split)
            num_sample=len(loader.dataset)
            num_classes = loader.dataset.lab_list.shape[-1]
            self.fea_bank=torch.randn(num_sample,256)
            self.score_bank = torch.randn(num_sample, num_classes).cuda()
            
            with torch.no_grad():
                for batch_id, (wavs, labels, filenames, indx) in enumerate(loader):
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                        h_features = self.upstream.model(wavs)
                        feas = self.featurizer.model(wavs, h_features)
                        # print(f"after featurizer: {len(feas)}{feas[0].shape}")
                        features_len = torch.IntTensor([len(feat) for feat in feas]).to(device=self.args.device)
                        # labels: assumed to be soft labels with shape (batch_size, num_classes)
                        # features: [batch, feat_dim]
                        feas = pad_sequence(feas, batch_first=True)
                        # print(f"after pad_sequence: {feas.shape}")
                        if "distillation" in self.args.downstream:
                            feas = self.downstream.model.student_projector(feas)
                        else:
                            feas = self.downstream.model.projector(feas)
                        # print(f"after projector: {feas.shape}")
                        if "distillation" in self.args.downstream:
                            logits, pooled, _  = self.downstream.model.student_model(feas, features_len)
                        else:
                            logits, pooled, _ = self.downstream.model.model(feas, features_len)      # [batch, num_classes]
                        # print(f"logits: {len(logits[0])}")
                        output_norm = F.normalize(pooled)
                        outputs = nn.Softmax(-1)(logits)
                        self.fea_bank[indx] = output_norm.detach().clone().cpu()
                        self.score_bank[indx] = outputs.detach().clone() #.cpu()
            if "distillation" in self.args.downstream:
                self.downstream.model.student_projector.train()
                self.downstream.model.student_model.train()
            else:
                self.downstream.model.projector.train()
                self.downstream.model.model.train()
            # self.downstream.model.projector.train()
            # self.upstream.model.train()
            self.featurizer.model.train()

        
        while pbar.n < pbar.total:
            try:
                if self.args.mi_based_sample_selection:
                    dataloader = self.downstream.model.get_train_dataloader(
                        mi_filtered=True,
                        featurizer_model=self.featurizer.model,
                        upstream_model=self.upstream.model,
                        args= self.args
                    )
                elif self.args.ent_mean_based_sample_selection:
                    dataloader = self.downstream.model.get_train_dataloader(
                        ent_mean_filtered=True,
                        featurizer_model=self.featurizer.model,
                        upstream_model=self.upstream.model,
                        args = self.args
                    )
                elif self.args.ent_each_based_sample_selection:
                    dataloader = self.downstream.model.get_train_dataloader(
                        ent_each_filtered=True,
                        featurizer_model=self.featurizer.model,
                        upstream_model=self.upstream.model,
                        args = self.args
                    )
                else:
                    dataloader = self.downstream.model.get_dataloader(train_split, epoch=epoch)
            except TypeError as e:
                if "unexpected keyword argument 'epoch'" in str(e):
                    dataloader = self.downstream.model.get_dataloader(train_split)
                    if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
                        dataloader.sampler.set_epoch(epoch)
                else:
                    raise
            #print(f"length dataloader: {len(dataloader)}")
            for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc='train', file=tqdm_file)):
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1
                    
                    if global_step == 1:
                        for split in self.config['runner']['eval_dataloaders']:
                            _ = self.evaluate(split, logger, global_step)


                    wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                    
                    wavs_stu = None
                    wavs_tea = None
                    if self.upstream.trainable and "distill" in self.args.downstream:
                        wavs_stu = self.stu_aug(wavs)[0]  # AddNoise returns list & len
                        wavs_tea = self.tea_aug(wavs)[0]

                    with torch.cuda.amp.autocast(enabled=amp):
                        if self.upstream.trainable and "distill" in self.args.downstream:
                            h_features = self.upstream.model(wavs_stu)
                            with torch.no_grad():
                                tea_features = self.additional_upstream.model(wavs_tea)
                        elif self.upstream.trainable:
                            h_features = self.upstream.model(wavs)
                        else:
                            with torch.no_grad():
                                # print(f"wavs: {len(wavs)}; {wavs[0].shape}")
                                h_features = self.upstream.model(wavs)
                        if wavs_stu:
                            # whether to other features copy without student_augmentation for other non-distill losses, e.g. NRC loss, IM loss
                            features = self.featurizer.model(wavs_stu, h_features)
                        else: 
                            features = self.featurizer.model(wavs, h_features)
                        
                        if self.additional_featurizers and self.upstream.trainable and "distill" in self.args.downstream:
                            addi_features = self.additional_featurizers.model(wavs_tea, tea_features)
                        elif self.additional_featurizers:
                            addi_features = self.additional_featurizers.model(wavs, h_features)

                        if specaug:
                            features, _ = specaug(features)
                            # if self.additional_featurizers and "distill" in self.args.downstream:
                            #     # AddNoise(noise_mean=0.0, noise_std=0.005, intensity=1.0)
                            #     addi_features, _ = add_noise(addi_features) ## strong augmentation?
                            if self.additional_featurizers:
                                addi_features, _ = specaug(addi_features)
                        if self.additional_featurizers:
                            if self.bank_store:
                                loss = self.downstream.model(
                                    train_split,
                                    features, *others,
                                    records = records,
                                    addi_features = addi_features,
                                    global_step = global_step,
                                    fea_bank = self.fea_bank,
                                    score_bank = self.score_bank,
                                    args = self.args
                                )
                            else:    
                                loss = self.downstream.model(
                                    train_split,
                                    features, *others,
                                    records = records,
                                    addi_features = addi_features,
                                    global_step = global_step,
                                    args = self.args,
                                    upstream_model = self.upstream.model,
                                    featurizer_model = self.featurizer.model,
                                )
                        else:
                            if self.bank_store:
                                loss = self.downstream.model(
                                    train_split,
                                    features, *others,
                                    records = records,
                                    global_step = global_step,
                                    fea_bank = self.fea_bank,
                                    score_bank = self.score_bank,
                                    args = self.args
                                )
                            else:
                                # mode, features, labels, filenames, records, global_step, upstream_model, featurizer_model, args
                                loss = self.downstream.model(
                                    train_split,
                                    features, *others,
                                    records = records,
                                    global_step = global_step,
                                    upstream_model = self.upstream.model,
                                    featurizer_model = self.featurizer.model,
                                    args = self.args
                                )
                                
                    batch_ids.append(batch_id)

                    gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                    loss = (loss / gradient_accumulate_steps)
                    if amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    del loss
                    

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        if is_initialized():
                            raise
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                # unscale
                if amp:
                    scaler.unscale_(optimizer)

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_paras, self.config['runner']['gradient_clipping'])

                # optimize
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                elif math.isnan(grad_norm):
                    print(f'[Runner] - grad norm is NaN at step {global_step}')
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()
                
                if self.upstream.trainable and "distill" in self.args.downstream:
                    ema_update(self.upstream.model, self.additional_upstream.model, 0.999)
                    ema_update(self.featurizer.model, self.additional_featurizers.model, 0.999)

                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    continue

                # logging
                if global_step % self.config['runner']['log_step'] == 0 or global_step==1:
                    self.downstream.model.log_records(
                        train_split,
                        records = records,
                        logger = logger,
                        global_step = global_step,
                        batch_ids = batch_ids,
                        total_batch_num = len(dataloader),
                    )
                    batch_ids = []
                    records = defaultdict(list)

                # evaluation and save checkpoint
                save_names = []

                if global_step % self.config['runner']['eval_step'] == 0:
                    for split in self.config['runner']['eval_dataloaders']:
                        save_names += self.evaluate(split, logger, global_step)

                if global_step % self.config['runner']['save_step'] == 0:
                    def check_ckpt_num(directory):
                        max_keep = self.config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)
                    save_names.append(f'states-{global_step}.ckpt')

                if len(save_names) > 0:
                    all_states = {
                        'Optimizer': optimizer.state_dict(),
                        'Step': global_step,
                        'Epoch': epoch,
                        'Args': self.args,
                        'Config': self.config,
                    }

                    for entry in self.all_entries:
                        if entry.trainable:
                            all_states[entry.name] = get_model_state(entry.model)

                    if scheduler:
                        all_states['Scheduler'] = scheduler.state_dict()

                    if is_initialized():
                        all_states['WorldSize'] = get_world_size()

                    save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                    tqdm.write(f'[Runner] - Save the checkpoint to:')
                    for i, path in enumerate(save_paths):
                        tqdm.write(f'{i + 1}. {path}')
                        torch.save(all_states, path)

                pbar.update(1)
            epoch += 1

        pbar.close()

        if self.args.push_to_hf_hub:
            self.push_to_huggingface_hub()
        if is_leader_process():
            logger.close()


    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in self.all_entries:
            trainings.append(entry.model.training)
            entry.model.eval()

        # prepare data
        dataloader = self.downstream.model.get_dataloader(split)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        batch_ids = []
        records = defaultdict(list)
        vis_feats = []
        vis_labs = []
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
            if batch_id > evaluate_steps:
                break

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            with torch.no_grad():
                h_features = self.upstream.model(wavs)
                features = self.featurizer.model(wavs, h_features)
                
                if self.additional_featurizers:
                    addi_features = self.additional_featurizers.model(wavs, h_features)
                    loss = self.downstream.model(
                        split,
                        features, *others,
                        records = records,
                        batch_id = batch_id,
                        addi_features = addi_features,
                        global_step = global_step,
                        args = self.args,
                        vis_feats = vis_feats,
                        vis_labs = vis_labs,
                    )
                else:
                    # mode, features, labels, filenames, records, global_step, upstream_model, featurizer_model, args,
                    if self.bank_store:
                        self.downstream.model(
                            split,
                            features, *others,
                            records = records,
                            global_step = global_step,
                            args = self.args,
                            vis_feats = vis_feats,
                            vis_labs = vis_labs,
                        )
                    else:
                        self.downstream.model(
                            split,
                            features, *others,
                            records = records,
                            batch_id = batch_id,
                            global_step = global_step,
                            upstream_model = self.upstream.model,
                            featurizer_model = self.featurizer.model,
                            args = self.args,
                            vis_feats = vis_feats,
                            vis_labs = vis_labs,
                        )
                batch_ids.append(batch_id)

        if global_step in {1, int(self.downstream.model.total_steps*0.05), int(self.downstream.model.total_steps*0.5), self.downstream.model.total_steps} and split == 'dev' and not self.args.downstream == "emotion_dev":
            vis_feats = torch.cat(vis_feats, dim=0)  # (N, D)
            vis_labs   = torch.cat(vis_labs, dim=0) 
            np.save(f"{self.downstream.model.expdir}/feats_step{global_step}.npy", vis_feats.numpy())
            np.save(f"{self.downstream.model.expdir}/labs_step{global_step}.npy", vis_labs.numpy())
            self.downstream.model.plot_feature_kde(vis_feats, vis_labs, save_prefix=f'step{global_step}')
        
        save_names = self.downstream.model.log_records(
            split,
            records = records,
            logger = logger,
            global_step = global_step,
            batch_ids = batch_ids,
            total_batch_num = len(dataloader),
        )
        batch_ids = []
        records = defaultdict(list)
        vis_feats = []
        vis_labs = []

        # prepare back to training
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        for entry, training in zip(self.all_entries, trainings):
            if training:
                entry.model.train().to(self.args.device)

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        return [] if type(save_names) is not list else save_names

    def inference(self):
        filepath = Path(self.args.evaluate_split)
        assert filepath.is_file(), filepath
        filename = filepath.stem

        if hasattr(self.downstream.model, "load_audio"):
            wav = self.downstream.model.load_audio(filepath)
        else:
            wav, sr = torchaudio.load(str(filepath))
            assert sr == SAMPLE_RATE, sr
        wavs = [wav.view(-1).to(self.args.device)]

        for entry in self.all_entries:
            entry.model.eval()

        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            self.downstream.model.inference(features, [filename])

    def push_to_huggingface_hub(self):
        """Creates a downstream repository on the Hub and pushes training artifacts to it."""
        if self.args.hf_hub_org.lower() != "none":
            organization = self.args.hf_hub_org
        else:
            organization = os.environ.get("HF_USERNAME")
        huggingface_token = HfFolder.get_token()
        print(f"[Runner] - Organisation to push fine-tuned model to: {organization}")
        
        # Extract upstream repository metadata
        if self.args.hub == "huggingface":
            model_info = HfApi().model_info(self.args.upstream, token=huggingface_token)
            downstream_model_id = model_info.sha
            # Exclude "/" characters from downstream repo ID
            upstream_model_id = model_info.modelId.replace("/", "__")
        else:
            upstream_model_id = self.args.upstream.replace("/", "__")
            downstream_model_id = str(uuid.uuid4())[:8]
        repo_name = f"{upstream_model_id}__{downstream_model_id}"
        # Create downstream repo on the Hub
        repo_url = HfApi().create_repo(
            token=huggingface_token,
            name=repo_name,
            organization=organization,
            exist_ok=True,
            private=False,
        )
        print(f"[Runner] - Created Hub repo: {repo_url}")

        # Download repo
        HF_HUB_DIR = "hf_hub"
        REPO_ROOT_DIR = os.path.join(self.args.expdir, HF_HUB_DIR, repo_name)
        REPO_TASK_DIR = os.path.join(REPO_ROOT_DIR, self.args.downstream, self.args.expname)
        print(f"[Runner] - Cloning Hub repo to {REPO_ROOT_DIR}")
        model_repo = Repository(
            local_dir=REPO_ROOT_DIR, clone_from=repo_url, use_auth_token=huggingface_token
        )
        # Pull latest changes if they exist
        model_repo.git_pull()

        # Copy checkpoints, tensorboard logs, and args / configs
        # Note that this copies all files from the experiment directory,
        # including those from multiple runs
        shutil.copytree(self.args.expdir, REPO_TASK_DIR, dirs_exist_ok=True, ignore=shutil.ignore_patterns(HF_HUB_DIR))

        # By default we use model.ckpt in the PreTrainedModel interface, so
        # rename the best checkpoint to match this convention
        checkpoints = list(Path(REPO_TASK_DIR).glob("*best*.ckpt"))
        if len(checkpoints) == 0:
            print("[Runner] - Did not find a best checkpoint! Using the final checkpoint instead ...")
            CKPT_PATH = (
                os.path.join(REPO_TASK_DIR, f"states-{self.config['runner']['total_steps']}.ckpt")
                )
        elif len(checkpoints) > 1:
            print(f"[Runner] - More than one best checkpoint found! Using {checkpoints[0]} as default ...")
            CKPT_PATH = checkpoints[0]
        else:
            print(f"[Runner] - Found best checkpoint {checkpoints[0]}!")
            CKPT_PATH = checkpoints[0]
        shutil.move(CKPT_PATH, os.path.join(REPO_TASK_DIR, "model.ckpt"))
        model_repo.lfs_track("*.ckpt")

        # Write model card
        self._create_model_card(REPO_ROOT_DIR)

        # Push everything to the Hub
        print("[Runner] - Pushing model files to the Hub ...")
        model_repo.push_to_hub()
        print("[Runner] - Training run complete!")
