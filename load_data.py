import numpy as np
import torch
from sklearn.utils import shuffle


def tkde_newdataset_normalization(path, random_seed):
    mat = np.load(path)
    data = mat['X']
    label = mat['y']
    label[label == 1] = int(-1)
    label[label == 0] = int(1)
    data_norm, label = shuffle(data, label, random_state=random_seed)
    data_norm = torch.Tensor(data_norm)
    label = torch.Tensor(label)
    data_list = []
    label_list = []
    data_number = data.shape[0]
    feature_number = data.shape[1]
    for i in range(10):
        data_list.append(data_norm[int(data_number * (0.1 * i)):int(data_number * (0.1 * (i + 1))), :int(feature_number * (0.1 * (i + 1)))])
        label_list.append(label[int(data_number * (0.1 * i)):int(data_number * (0.1 * (i + 1)))])

    return data_list, label_list