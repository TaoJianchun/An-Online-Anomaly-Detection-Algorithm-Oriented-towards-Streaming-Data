
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from load_data import *
from graph_generation import graph_generation, graph_incremental
from hypersphere import HyperSphere_Construction
from metric import mask_list
from parameter import *


import numpy as np
import os
import torch
import numpy as np
import random
import os

def set_seed(seed=1020):
    # 固定 Python 内部随机种子
    random.seed(seed)
    # 固定 Numpy 随机种子
    np.random.seed(seed)
    # 固定 PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    # 设置 CuDNN 后端为确定性模式（牺牲一点点速度，换取绝对复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(data_list, label_list,model_layers, training_epoch, reduced_size, label_rate, path):

    truth_label = []
    Starter = HyperSphere_Construction(reduced_size=reduced_size,layer_num=model_layers, Epoch=training_epoch)

    for times, data in enumerate(data_list):

        if times == 0:

            label = label_list[times]

            mask_data = mask_list(data, label_rate)
            sample_num, feature_num = data.size()

            graph_data = graph_generation(data, reduced_size, label)
            truth_label.extend(graph_data.y.numpy())
            Starter.First_Construction(graph_data = graph_data, mask = mask_data, feature_num = feature_num, data = data)

        else:

            label = label_list[times]
            mask_data = mask_list(data, label_rate)
            old_sample_num, old_feature_num = sample_num, feature_num
            sample_num, feature_num = data.size()
            graph_data = graph_incremental(graph_data.x[:old_feature_num].detach().cpu(), data, label, feature_num - old_feature_num, reduced_size)
            truth_label.extend(graph_data.y.numpy())
            Starter.Second_Construction(graph_data=graph_data, mask=mask_data, feature_num= feature_num, old_feature_num=old_feature_num)
    auc = roc_auc_score(y_true=np.array(truth_label), y_score=np.array(Starter.predictionlist))

    print('Acc ={:.4f}, F1 ={:.4f}, Recall={:.4f}, Precision={:.4f}, AUC-ROC={:.4f}'.format(
        accuracy_score(y_true=np.array(truth_label), y_pred=np.array(Starter.predictionlist)),
        f1_score(y_true=np.array(truth_label), y_pred=np.array(Starter.predictionlist), pos_label=0),
        recall_score(y_true=np.array(truth_label), y_pred=np.array(Starter.predictionlist), pos_label=0),
        precision_score(y_true=np.array(truth_label), y_pred=np.array(Starter.predictionlist), pos_label=0),  
        auc
    ))

    return Starter





if __name__ == '__main__':

    set_seed(1020)
    random_seed = 1020

    f1_list = []
    precision = []
    recall = []
    data_path = "data/3_backdoor.npz"
    data_list, label_list = tkde_newdataset_normalization(data_path,random_seed=1020)

    model_layers = 2
    training_epoch = 100
    reduced_size = return_reduced_size('backdoor')
    label_rate = 0.2

Starter = train(data_list, label_list, model_layers, training_epoch, reduced_size, label_rate, path='backdoor')

# ==================== 杀手锏：概念漂移可视化图表 (修复版) ====================
import matplotlib.pyplot as plt
import numpy as np

# 获取记录的漂移分数
drift_scores = Starter.drift_history

if len(drift_scores) > 0:
    # 设置图表大小和清晰度
    plt.figure(figsize=(12, 6), dpi=100)

    # 1. 绘制模型输出的漂移得分曲线 (蓝色)
    plt.plot(drift_scores, label='Concept Uncertainty Score', color='royalblue', linewidth=2, marker='o', markersize=5)

    # 2. 计算动态自适应阈值 (mu + 2*sigma)
    thresholds = [np.nan, np.nan]  # 前两个点没有足够历史窗口
    for i in range(2, len(drift_scores)):
        window = drift_scores[max(0, i - 5):i]
        mu = np.mean(window)
        sigma = np.std(window)
        thresholds.append(mu + 2 * sigma)

    # 绘制动态阈值曲线 (橙色虚线)
    plt.plot(thresholds, label='Dynamic Threshold ($\mu + 2\sigma$)', color='darkorange', linestyle='--', linewidth=2)

    # 3. 标记发生概念漂移、触发模型演化的高光时刻 (红色大圆点)
    # 【修复点】：使用 label_added 标志位，确保图例 (Legend) 中只出现一次标签，避免报错
    label_added = False
    for i in range(2, len(drift_scores)):
        if drift_scores[i] > thresholds[i] and drift_scores[i] > 0.05:
            if not label_added:
                plt.scatter(i, drift_scores[i], color='crimson', s=150, zorder=5,
                            label='Concept Drift Detected & Adapted!')
                label_added = True
            else:
                plt.scatter(i, drift_scores[i], color='crimson', s=150, zorder=5)

            # 画一条垂直辅助线，标明概念漂移发生的时间点
            plt.axvline(x=i, color='gray', linestyle=':', alpha=0.5)

    # 4. 图表排版美化 (按照顶会论文图表标准)
    plt.title("Dynamic Concept Drift Adaptation over Streaming Data", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Streaming Data Blocks (Time)", fontsize=14)
    plt.ylabel("Concept Uncertainty (Reconstruction Error)", fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 优化坐标轴刻度和边距
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    # 保存为高清图片并显示
    plt.savefig("concept_drift_visualization.png", dpi=300)
    print("\n✅ 概念漂移可视化图表已生成并保存为 concept_drift_visualization.png")
    plt.show()

    # ==================== 杀手锏图表二：特征稀疏率变化图 ====================
    sparsity_scores = Starter.sparsity_history

    if len(sparsity_scores) > 0:
        plt.figure(figsize=(10, 5), dpi=100)

        # 绘制稀疏率曲线并填充下方阴影 (使用清新的海洋绿)
        plt.plot(sparsity_scores, label='Feature Sparsity Ratio (%)', color='seagreen', linewidth=2.5, marker='s',
                 markersize=6)
        plt.fill_between(range(len(sparsity_scores)), sparsity_scores, color='seagreen', alpha=0.15)

        # 图表排版美化
        plt.title("Dynamic Feature Elimination via $\ell_{1,2}$-norm Regularization", fontsize=15, fontweight='bold',
                  pad=15)
        plt.xlabel("Streaming Data Blocks (Time)", fontsize=13)

        # 将 Y 轴转换为百分比显示
        import matplotlib.ticker as ticker

        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        plt.ylabel("Sparsity Ratio (Eliminated Features)", fontsize=13)

        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # 保存为高清图片并显示
        plt.savefig("feature_sparsity_visualization.png", dpi=300)
        print("✅ 特征稀疏率可视化图表已生成并保存为 feature_sparsity_visualization.png")
        plt.show()
    # ========================================================================



