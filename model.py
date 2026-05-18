from torch import nn
from torch_geometric.graphgym import GCNConv
from torch_geometric.nn import GATConv, SAGEConv, SGConv

from graph_generation import *
import torch.nn
import torch.nn.functional as F


class GraphModel(nn.Module):
    def __init__(self, layer_num, in_features, hidden_features, edge_index_size, num_classes,graphconv='SAGE'):
        super(GraphModel, self).__init__()
        self.convs = nn.ModuleList()
        self.edge_weight = nn.Parameter(torch.ones(edge_index_size[1], dtype=torch.float))
        in_features_copy = in_features
        for i in range(layer_num - 1):
            if graphconv == 'GCN':
                self.convs.append(GCNConv(in_features, hidden_features))
            elif graphconv == 'SAGE':
                self.convs.append(SAGEConv(in_features, hidden_features))
            elif graphconv == 'SGC':
                self.convs.append(SGConv(in_features, hidden_features))
            elif graphconv == 'GAT':
                self.convs.append(GATConv(in_features, hidden_features))
            else:
                raise NotImplementedError

            in_features = hidden_features  # 更新输入特征维度


        # 最后一层调整为输出类别数
        if graphconv == 'GCN':
            self.convs.append(GCNConv(hidden_features, in_features_copy))
            self.convs.append(GCNConv(in_features_copy, num_classes))
        elif graphconv == 'SAGE':
            self.convs.append(SAGEConv(hidden_features, in_features_copy))
            self.convs.append(SAGEConv(in_features_copy, num_classes))

        elif graphconv == 'SGC':
            self.convs.append(SGConv(hidden_features, in_features_copy))
            self.convs.append(SGConv(in_features_copy, num_classes))

        elif graphconv == 'GAT':
            self.convs.append(GATConv(hidden_features, in_features_copy))
            self.convs.append(GATConv(in_features_copy, num_classes))

        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(hidden_features) for _ in range(layer_num - 1)])
        self.bn = nn.BatchNorm1d(in_features)

    def forward(self, graph_data, feature_num):
        x, edge_index = graph_data.x, graph_data.edge_index
        for i, conv in enumerate(self.convs[:-2]):
            if isinstance(conv, (GCNConv, GATConv)):
                x = conv( x, edge_index)
            else:
                x = conv( x, edge_index)
            x = self.batchnorm[i]( x)
            x = nn.functional.relu( x)


        x = self.convs[-2](x, edge_index)
        x = nn.functional.relu(x)

        output = self.convs[-1](x, edge_index)

        return  x[:feature_num],  x[feature_num:], output[feature_num:]



