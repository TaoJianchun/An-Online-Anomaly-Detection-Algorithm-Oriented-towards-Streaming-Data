import numpy as np
import torch
from torch_geometric.data import Data


def create_data(data, embedding_matrix):
    # 使用矩阵乘法加速
    rec_samples = torch.matmul(data, embedding_matrix)
    return rec_samples.tolist()


def create_node(data, reduced_size):
    sample_num, feature_num = data.shape
    # 直接在 torch 中生成随机矩阵，避免 numpy 转换开销
    W = torch.rand(feature_num, reduced_size, dtype=torch.float32)

    # 核心加速：摒弃双层 for 循环，使用矩阵乘法完成重构
    # data: [sample_num, feature_num], W: [feature_num, reduced_size]
    # 结果 reconstructed_data: [sample_num, reduced_size]
    reconstructed_data = torch.matmul(data, W)

    node_feature = torch.cat((W, reconstructed_data), dim=0)
    return node_feature


def create_edge(x, feature_num):
    init_source, init_end = torch.nonzero(x).transpose(0, 1)

    edge_source = init_source + feature_num
    edge_end = init_end
    edge_source_ = torch.cat([edge_source, edge_end], dim=0)

    edge_end_ = torch.cat([edge_end, edge_source], dim=0)
    edge_index = torch.stack([edge_source_, edge_end_], dim=0)  # [2, edge_num]
    return edge_index


def create_label(label):
    # 向量化标签生成，把 == 1 的变成 1，其他变成 0
    y = torch.where(label == 1, torch.tensor(1.0), torch.tensor(0.0))
    return y


def graph_generation(x, reduced_size, y):
    sample_num, feature_num = x.size()
    node_feature = create_node(x, reduced_size)

    edge_index = create_edge(x, feature_num)
    y = create_label(y)
    data = Data(x=node_feature, edge_index=edge_index, y=y)
    return data


def graph_incremental(old_embedding_vectors, new_samples, new_label, new_feature_num, reduced_size):
    sample_num, feature_num = new_samples.shape
    new_graph_edge = create_edge(new_samples, feature_num)
    new_label = create_label(new_label)

    if new_feature_num != 0:
        new_embedding_vectors = torch.rand(new_feature_num, reduced_size, dtype=torch.float32)
        embedding_matrix = torch.cat((old_embedding_vectors, new_embedding_vectors), dim=0)
    else:
        embedding_matrix = old_embedding_vectors.cpu()

    # 核心加速：使用矩阵乘法替代增量更新时的双层 for 循环
    reconstructed_data = torch.matmul(new_samples, embedding_matrix)

    node_feature = torch.cat((embedding_matrix, reconstructed_data), dim=0)

    data = Data(x=node_feature, edge_index=new_graph_edge, y=new_label)
    return data