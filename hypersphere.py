from model import GraphModel
import torch
from torch import nn
import numpy as np  # 方向二需要用到 np.mean 和 np.std


# ================= 创新方向一：L1,2 混合范数计算函数 =================
def l1_2_regularization(weight_matrix):
    """
    计算特征嵌入矩阵的 L1,2 混合范数，促使无用特征的权重整行稀疏化
    """
    l2_norms = torch.norm(weight_matrix + 1e-8, p=2, dim=1)
    l1_2_norm = torch.sum(l2_norms)
    return l1_2_norm


# ====================================================================

class HyperSphere_Construction:
    def __init__(self, reduced_size=20, layer_num=3, Epoch=30, rec_weight=1):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reduced_size = reduced_size
        self.layer_num = layer_num
        self.MSELoss = nn.MSELoss() # 用于特征重构与超球体距离计算
        self.Epoch = Epoch
        self.rec_weight = rec_weight
        self.predictionlist = []

        # ================= 创新方向二：METER 漂移检测历史记录 =================
        self.drift_history = []

        self.sparsity_history = []
        # =====================================================================

    def First_Construction(self, graph_data, mask, feature_num, data):
        edge_index_size = graph_data.edge_index.size()
        self.model = GraphModel(layer_num=self.layer_num, in_features=self.reduced_size, hidden_features=64,
                                edge_index_size=edge_index_size,
                                num_classes=2).to(self.device)

        self.mask = mask
        self.graph_data = graph_data.to(self.device)

        # 优化器设置
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

        self.masklabel = graph_data.y.clone().detach()
        self.masklabel[~self.mask] = -1

        class_counts = torch.bincount(self.masklabel[self.mask].long())
        weights = len(self.masklabel[self.mask]) / (class_counts * len(class_counts)).to(self.device)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weights)

        # 【修复点3】：调大稀疏惩罚系数，0.0001 太小了，建议从 0.01 或 0.1 起步
        lambda_sparse = 0.01

        for _ in range(self.Epoch):
            W, X, Pred = self.model(graph_data.detach(), feature_num)

            # 提取正常样本的表示 (假设 self.masklabel == 1 代表正常样本，0代表已知异常)
            normal_mask = (self.masklabel[self.mask] == 1) | (self.masklabel[self.mask] == -1)
            normal_X = X[mask][normal_mask]

            # 【修复点1】：正确计算超球体中心和半径
            self.center = torch.mean(normal_X, dim=0)  # 球心：正常样本的均值向量

            # 计算所有样本到球心的欧氏距离
            dist = torch.sqrt(torch.sum((X - self.center) ** 2, dim=1))

            # 真实半径：正常样本到球心的最大距离
            normal_dist = torch.sqrt(torch.sum((normal_X - self.center) ** 2, dim=1))
            self.radius = torch.max(normal_dist) if len(normal_dist) > 0 else torch.tensor(0.1).to(self.device)

            optimizer.zero_grad()

            # 【修复点2】：正确计算内部和外部 Loss (使用 torch.clamp 避免降维错误)
            # 对于正常样本：如果 dist > radius，受到惩罚 (拉回球内)
            inside_dist_diff = dist[(self.masklabel == 1) | (self.masklabel == -1)] - self.radius
            inside_loss = torch.mean(torch.clamp(inside_dist_diff, min=0.0))

            # 对于已知异常样本：如果 dist < radius，受到惩罚 (推出球外)
            anomaly_dist_diff = self.radius - dist[self.masklabel == 0]
            if len(anomaly_dist_diff) > 0:
                outside_loss = torch.mean(torch.clamp(anomaly_dist_diff, min=0.0))
            else:
                outside_loss = torch.tensor(0.0).to(self.device)

            dist_loss = inside_loss + outside_loss

            # 监督分类 Loss
            pred_loss = self.CrossEntropyLoss(Pred[self.mask], self.graph_data.y[self.mask].long())

            # 稀疏惩罚项 (确保上一轮交流中 l1_2_regularization 里用的是 torch.sum)
            sparse_penalty = lambda_sparse * l1_2_regularization(W)

            # 权重融合：平衡距离损失、预测损失与稀疏约束
            total_loss = 1.0 * dist_loss + 1.0 * pred_loss + sparse_penalty

            total_loss.backward(retain_graph=True)
            optimizer.step()

            pred = torch.argmax(Pred, dim=1)

        self.predictionlist.extend(pred.detach().cpu().numpy())

        # ================= 记录当前数据块的特征稀疏率 =================
        with torch.no_grad():
            row_norms = torch.norm(W, p=2, dim=1)
            zero_threshold = 1e-4
            dead_features_count = torch.sum(row_norms < zero_threshold).item()
            sparsity_ratio = dead_features_count / feature_num
            self.sparsity_history.append(sparsity_ratio)
            # 可以打印出来看看是否起作用了
            # print(f"Initial Epoch finished. Sparsity Ratio: {sparsity_ratio:.2%}")
        # ====================================================================

    def Second_Construction(self, graph_data, mask, feature_num, old_feature_num):
        self.mask = mask
        self.graph_data = graph_data.to(self.device)
        self.masklabel = graph_data.y.clone().detach()
        self.masklabel[~self.mask] = -1

        class_counts = torch.bincount(self.masklabel[self.mask].long())
        weights = len(self.masklabel[self.mask]) / (class_counts * len(class_counts)).to(self.device)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weights)

        # 初始默认参数
        current_lr = 0.001
        current_rec_weight = self.rec_weight
        drift_score = 0.0

        # ================= 1. METER 概念漂移检测 (评估模式) =================
        try:
            self.model.eval()
            with torch.no_grad():
                _, X_temp, _ = self.model(graph_data.detach(), feature_num)

                normal_mask = (self.masklabel[self.mask] == 1) | (self.masklabel[self.mask] == -1)
                normal_X_temp = X_temp[mask][normal_mask]

                if len(normal_X_temp) > 0:
                    center_temp = self.center if hasattr(self, 'center') else torch.mean(normal_X_temp, dim=0)
                    dist_temp = torch.sqrt(torch.sum((normal_X_temp - center_temp) ** 2, dim=1))

                    radius_temp = torch.quantile(dist_temp, 0.95) if len(dist_temp) > 10 else torch.max(dist_temp)
                    drift_score = torch.mean(torch.clamp(dist_temp - radius_temp, min=0.0)).item()
                else:
                    drift_score = 0.0
        except Exception as e:
            print(f"Drift score calculation error: {e}")
            pass

        # ================= 2. 智能演化控制与阈值抗中毒 =================
        if len(self.drift_history) >= 5:
            hist_mean = np.mean(self.drift_history[-5:])
            hist_std = np.std(self.drift_history[-5:]) + 1e-4

            if drift_score > hist_mean + 3 * hist_std and drift_score > 0.001:

                print(
                    f"⚠️ [METER 智能控制] 检测到概念漂移! 得分: {drift_score:.4f} (动态阈值: {hist_mean + 2 * hist_std:.4f})")
                current_lr = 0.005
                current_rec_weight = 0.0
            else:
                print(f"✅ 数据流概念平稳. 得分: {drift_score:.4f} (动态阈值: {hist_mean + 2 * hist_std:.4f})")
                self.drift_history.append(drift_score)  # 平稳期才加入历史
        else:
            self.drift_history.append(drift_score)
            print(f"⏳ 收集漂移历史... 当前得分: {drift_score:.4f}")

        # ================= 3. 动态模型更新 (训练模式) =================
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr, weight_decay=1e-4)
        W_old = graph_data.x[:feature_num]

        lambda_sparse = 0.05

        streaming_epoch = 1

        for _ in range(streaming_epoch):
            optimizer.zero_grad()  # 确保每次迭代前清空梯度

            W, X, Pred = self.model(graph_data.detach(), feature_num)

            rec_loss = current_rec_weight * self.MSELoss(W, W_old.detach())
            W_old = W.detach()

            normal_mask = (self.masklabel[self.mask] == 1) | (self.masklabel[self.mask] == -1)
            normal_X = X[mask][normal_mask]

            if len(normal_X) > 0:
                self.center = torch.mean(normal_X, dim=0)
                dist_to_center = torch.sqrt(torch.sum((normal_X - self.center) ** 2, dim=1))
                self.radius = torch.quantile(dist_to_center, 0.99) if len(dist_to_center) > 1 else torch.max(
                    dist_to_center)

            dist = torch.sqrt(torch.sum((X - self.center) ** 2, dim=1))

            inside_dist_diff = dist[(self.masklabel == 1) | (self.masklabel == -1)] - self.radius
            inside_loss = torch.mean(torch.clamp(inside_dist_diff, min=0.0))

            anomaly_dist_diff = self.radius - dist[self.masklabel == 0]
            if len(anomaly_dist_diff) > 0:
                outside_loss = torch.mean(torch.clamp(anomaly_dist_diff, min=0.0))
            else:
                outside_loss = torch.tensor(0.0).to(self.device)

            dist_loss = inside_loss + outside_loss
            pred_loss = self.CrossEntropyLoss(Pred[self.mask], self.graph_data.y[self.mask].long())

            sparse_penalty = lambda_sparse * l1_2_regularization(W)

            # 大一统 Loss
            total_loss = 1.0 * dist_loss + 1.0 * pred_loss + sparse_penalty + rec_loss

            # 【终极防崩溃安全锁】
            if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                total_loss.backward(retain_graph=True)
                optimizer.step()
            else:
                pass

            pred = torch.argmax(Pred, dim=1)

        # 循环结束，记录预测
        self.predictionlist.extend(pred.detach().cpu().numpy())

        # ================= 4. 记录与打印当前数据块的特征稀疏率 =================
        with torch.no_grad():  # 这里的 no_grad 是绝对正确的，因为它只用于统计特征存活率
            row_norms = torch.norm(W, p=2, dim=1)
            zero_threshold = 5e-4
            dead_features_count = torch.sum(row_norms < zero_threshold).item()
            sparsity_ratio = dead_features_count / feature_num
            self.sparsity_history.append(sparsity_ratio)

            active_features = feature_num - dead_features_count
            print(
                f"📉 [SAD 稀疏化监控] 当前特征空间: 活跃 {active_features}/{feature_num} | 实时特征稀疏率: {sparsity_ratio:.2%}")
        # ====================================================================

    def load_model(self, model):
        state_dict_load = torch.load(self.path_state_dict)
        model.load_state_dict(state_dict_load)
        return model