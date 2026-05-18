import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告


def mask_list(data, ratio):
    data_idx = torch.arange(0, data.size(0))
    train_mask = torch.zeros(data.size(0), dtype=torch.bool)
    train_idx = data_idx[:int(data.size(0) * ratio)]
    train_mask[train_idx] = True
    return train_mask
