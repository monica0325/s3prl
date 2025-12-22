# -*- coding: utf-8 -*-
"""
    Refactored distillation dataset code,
    模仿 simp/dataset.py 的結構，並保留原本的知識（如 WavExtractor, WavSet 等功能）。
"""

import json
import glob
import os
import csv
import pickle as pk
import numpy as np
import torch
import torch.utils.data as torch_utils
import librosa
from tqdm import tqdm
from multiprocessing import Pool

# 保留與原 distillation 相同的函數/類別
SAMPLE_RATE = 16000

def get_norm_stat_for_wav(wav_list, verbose=False):
    count = 0
    wav_sum = 0
    wav_sqsum = 0
    iterator = tqdm(wav_list) if verbose else wav_list
    for cur_wav in iterator:
        wav_sum += np.sum(cur_wav)
        wav_sqsum += np.sum(cur_wav**2)
        count += len(cur_wav)
    
    wav_mean = wav_sum / count
    wav_var = (wav_sqsum / count) - (wav_mean**2)
    wav_std = np.sqrt(wav_var)
    return wav_mean, wav_std

def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    return raw_wav

class WavExtractor:
    """
    與原 distillation/dataset.py 保持同樣功能，採用多進程讀取。
    """
    def __init__(self, wav_paths, nj=24):
        self.wav_path_list = wav_paths
        self.nj = nj

    def extract(self):
        print("Extracting wav files...")
        with Pool(self.nj) as p:
            wav_list = list(tqdm(p.imap(extract_wav, self.wav_path_list), total=len(self.wav_path_list)))
        return wav_list

class WavSet(torch_utils.Dataset):
    """
    與 simp/dataset.py 中的 WavSet 類似，但保留 distillation 的細節 (lab_type='categorical' 等)。
    """
    def __init__(self, wav_list, lab_list, utt_list, print_dur=False, lab_type='categorical',
                 wav_mean=0., wav_std=1., label_config=None):
        super(WavSet, self).__init__()
        self.wav_list = wav_list
        self.lab_list = lab_list
        self.utt_list = utt_list
        self.print_dur = print_dur
        self.lab_type = lab_type
        self.label_config = label_config

        self.wav_mean = wav_mean
        self.wav_std = wav_std
        self.max_dur = 10 * SAMPLE_RATE

        # 如果尚未指定，計算一次整體 mean, std
        if self.wav_mean is None or self.wav_std is None:
            self.wav_mean, self.wav_std = get_norm_stat_for_wav(self.wav_list)

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = extract_wav(self.wav_list[idx])[:self.max_dur]
        cur_dur = len(cur_wav)
        cur_wav = (cur_wav - self.wav_mean) / (self.wav_std + 1e-6)
        cur_utt = self.utt_list[idx]

        if self.lab_type == "categorical":
            cur_lab = self.lab_list[idx]
        else:
            cur_lab = None  # distillation 原程式只支援 categorical

        if self.print_dur:
            return (cur_wav, cur_lab, cur_utt, idx, cur_dur)
        else:
            return (cur_wav, cur_lab, cur_utt, idx) # wav discrete sample list, emo_dict, label

def collate_fn_padd(batch):
    """
    與 simp/dataset.py 保持相同的回傳形式，
    (list_of_wavTensor, tensor_of_labels, list_of_utt)。
    """
    total_wav = []
    total_lab = []
    total_utt = []
    total_idx = []

    # 如果 print_dur=True，則 batch 內容會多一個 dur
    # 不過這裡我們只處理預設情況
    for entry in batch:
        if len(entry) == 5:
            wav, lab, utt, idx, _ = entry
        else:
            wav, lab, utt, idx = entry
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_utt.append(utt)
        total_idx.append(idx)

    total_lab = torch.Tensor(np.asarray(total_lab))
    return total_wav, total_lab, total_utt, total_idx

# -------------------------------------------------------------------
# DataManager & prepare_datasets: 
# 模仿 simp 的做法，將 dataset 初始化流程集中在一個函式裡
# -------------------------------------------------------------------
class DataManager:
    """
    與原 distillation/dataset.py 之中的 DataManager 類似，
    但加上與 simp/dataset.py 相同的介面：get_wav_path, get_utt_list, get_msp_labels。
    """
    def __init__(self, env_path):
        self.env_dict = self.__load_env__(env_path)
        self.msp_label_dict = None

    def __load_env__(self, env_path):
        with open(env_path, 'r') as f:
            return json.load(f)

    def get_wav_path(self, split_type=None, wav_loc=None, lbl_loc=None):
        if split_type is None:
            wav_list = glob.glob(os.path.join(wav_loc, "*.wav"))
        else:
            utt_list = self.get_utt_list(split_type, lbl_loc)
            wav_list = [os.path.join(wav_loc, utt_id) for utt_id in utt_list]
        wav_list.sort()
        return wav_list

    def get_utt_list(self, split_type, lbl_loc):
        label_path = lbl_loc
        utt_list = []
        sid = self.env_dict["data_split_type"][split_type]
        with open(label_path, 'r') as f:
            f.readline()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                stype = row[-1] #[6] #-1
                if stype == sid:
                    utt_list.append(utt_id) # i-th wav data sample with utt_id
        utt_list.sort()
        return utt_list

    def __load_msp_cat_label_dict__(self, lbl_loc):
        self.msp_label_dict = dict()
        emo_class_list = self.get_categorical_emo_class()
        with open(lbl_loc, 'r') as f:
            header = f.readline().split(",")
            header = [col.strip() for col in header]
            print(f"header: {header}")
            emo_idx_list = []
            for emo_class in emo_class_list:
                emo_idx_list.append(header.index(emo_class))
            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                cur_emo_lab = []
                for emo_idx in emo_idx_list:
                    cur_emo_lab.append(float(row[emo_idx])) # emotion label distribution of utt_id-th sample
                self.msp_label_dict[utt_id] = cur_emo_lab

    def get_msp_labels(self, utt_list, lab_type='categorical', lbl_loc=None):
        if lab_type == "categorical":
            if self.msp_label_dict is None:
                self.__load_msp_cat_label_dict__(lbl_loc)
            return np.array([self.msp_label_dict[utt_id] for utt_id in utt_list])
        else:
            raise NotImplementedError("Only 'categorical' label type is implemented.")

    def get_categorical_emo_class(self):
        return self.env_dict["categorical"]["emo_type"]

    def get_label_config(self, label_type):
        assert label_type in ["categorical", "dimensional"]
        return self.env_dict[label_type]


def prepare_datasets(datarc, config_path):
    """
    與 simp/dataset.py 類似，回傳:
      train_dataset, dev_dataset, test_dataset,
      class_balanced_weights, k_threshold, categorical_emo
    """
    dam = DataManager(config_path)

    audio_path = os.path.join(datarc['root'], datarc['corpus'], "Audios")
    label_path = os.path.join(
        datarc['root'],
        datarc['corpus'],
        datarc['p_or_s'],
        "labels_consensus_" + datarc['test_fold'].replace("fold", "") + ".csv"
        # datarc['root'],
        # datarc['corpus'],
        # datarc['p_or_s'],
        # datarc['src'],
        # "labels_consensus_" + datarc['test_fold'].replace("fold", "") + ".csv"
    )
    print(f"total label_path: {label_path}")
    # 取得資料集的 utterance ids
    train_utts = dam.get_utt_list("train", lbl_loc=label_path)
    print(f"train_utts: {len(train_utts)}")
    dev_utts = dam.get_utt_list("dev", lbl_loc=label_path)
    print(f"dev_utts: {len(dev_utts)}")
    test_utts = dam.get_utt_list("test", lbl_loc=label_path)
    print(f"test_utts: {len(test_utts)}")
    
    # 取得音檔路徑
    train_wav_paths = dam.get_wav_path("train", wav_loc=audio_path, lbl_loc=label_path)
    print(f"train_wav_paths: {train_wav_paths[0]}")
    dev_wav_paths = dam.get_wav_path("dev", wav_loc=audio_path, lbl_loc=label_path)
    test_wav_paths = dam.get_wav_path("test", wav_loc=audio_path, lbl_loc=label_path)

    # 取得情緒標籤
    train_labs = dam.get_msp_labels(train_utts, lab_type='categorical', lbl_loc=label_path)
    print(f"train_labs: {list(train_labs[0])}")
    dev_labs = dam.get_msp_labels(dev_utts, lab_type='categorical', lbl_loc=label_path)
    test_labs = dam.get_msp_labels(test_utts, lab_type='categorical', lbl_loc=label_path)

    # 計算 class balanced weights
    k_threshold = 1 / train_labs.shape[1]
    train_labs_PT = torch.Tensor(train_labs)
    train_labs_binary_PT = torch.where(train_labs_PT > k_threshold, 1.0, 0.0)
    samples_per_cls = torch.sum(train_labs_binary_PT, dim=0)

    beta = (train_labs.shape[0]-1) / train_labs.shape[0]
    no_of_classes = train_labs.shape[1]
    effective_num = 1.0 - torch.pow(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    class_balanced_weights = weights / torch.sum(weights) * no_of_classes
    print(f"class_balanced_weights: {class_balanced_weights.shape}")

    # 載入或計算音檔平均/標準差
    train_wavs_np_path = os.path.join(
        datarc['root'],
        datarc['corpus'],
        datarc['p_or_s'],
        "Train_wavs_numpy_" + datarc['test_fold'] + ".pkl"
    )
    if not os.path.exists(train_wavs_np_path):
        print("Saving Wavs Numpy files:", train_wavs_np_path)
        train_wavs = WavExtractor(train_wav_paths).extract()
        wav_mean, wav_std = get_norm_stat_for_wav(train_wavs)
        stats = {"wav_mean": wav_mean, "wav_std": wav_std}
        with open(train_wavs_np_path, 'wb') as f:
            pk.dump(stats, f)
    else:
        with open(train_wavs_np_path, 'rb') as f:
            stats = pk.load(f)
        wav_mean = stats["wav_mean"]
        wav_std = stats["wav_std"]

    # Label Config
    label_config = dam.get_label_config(label_type='categorical')

    # 建立 dataset
    train_dataset = WavSet(
        train_wav_paths,
        train_labs,
        train_utts,
        print_dur=True,
        lab_type='categorical',
        label_config=label_config,
        wav_mean=wav_mean,
        wav_std=wav_std
    )
    dev_dataset = WavSet(
        dev_wav_paths,
        dev_labs,
        dev_utts,
        print_dur=True,
        lab_type='categorical',
        label_config=label_config,
        wav_mean=wav_mean,
        wav_std=wav_std
    )
    test_dataset = WavSet(
        test_wav_paths,
        test_labs,
        test_utts,
        print_dur=True,
        lab_type='categorical',
        label_config=label_config,
        wav_mean=wav_mean,
        wav_std=wav_std
    )

    # 回傳
    categorical_emo = dam.get_categorical_emo_class()
    return train_dataset, dev_dataset, test_dataset, class_balanced_weights, k_threshold, categorical_emo
