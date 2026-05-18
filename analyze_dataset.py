import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 设置中文字体，防止图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def analyze_and_visualize(file_path):
    print("正在加载数据，请稍候...")
    # 1. 加载数据 (根据你的实际情况，假设是 npz 或包含了 'X' 和 'y' 的 npy)
    try:
        mat = np.load(file_path)
        data = mat['X']
        label = mat['y']
    except Exception as e:
        print(f"数据加载失败，请检查路径。错误信息: {e}")
        return

    # ==========================================
    # 模块一：全局基础统计
    # ==========================================
    sample_num, feature_num = data.shape
    anomaly_ratio = np.sum(label == 1) / len(label)

    print("\n" + "=" * 30)
    print(f"📊 全局数据集统计信息")
    print(f"总样本数 (Total Samples): {sample_num}")
    print(f"总特征数 (Total Features): {feature_num}")
    print(f"异常样本占比 (Anomaly Ratio): {anomaly_ratio:.2%}")
    print("=" * 30)

    # ==========================================
    # 模块二：流式开放空间 (OFS) 维度膨胀可视化
    # 这个图可以完美展示你是如何把静态数据变成“流”的
    # ==========================================
    blocks = 10
    # 模拟你代码里的切分逻辑
    features_per_block = [int(feature_num * (0.1 * (i + 1))) for i in range(blocks)]
    samples_per_block = [int(sample_num * 0.1)] * blocks

    plt.figure(figsize=(10, 5))
    # 使用阶梯图 (step) 来表现块状数据的跃迁
    plt.step(range(1, blocks + 1), features_per_block, where='post', marker='o', color='#2ca02c', linewidth=2.5)
    plt.fill_between(range(1, blocks + 1), features_per_block, step="post", alpha=0.2, color='#2ca02c')

    plt.title('开放特征空间 (OFS) 维度膨胀模拟过程', fontsize=15)
    plt.xlabel('数据块序号 (Time Step / Block)', fontsize=12)
    plt.ylabel('系统暴露的特征维度数', fontsize=12)
    plt.xticks(range(1, blocks + 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 模块三：概念漂移 (Concept Drift) 降维对比图
    # 提取 Block 1 (初期) 和 Block 8 (后期，有漂移) 对比
    # ==========================================
    print("\n正在进行 PCA 降维以可视化数据分布，这可能需要几十秒...")

    # 提取 Block 1 (只有前 10% 的特征和样本)
    b1_data = data[:int(sample_num * 0.1), :int(feature_num * 0.1)]
    b1_label = label[:int(sample_num * 0.1)]

    # 提取 Block 8 (包含前 80% 的特征，发生漂移的阶段)
    b8_data = data[int(sample_num * 0.7):int(sample_num * 0.8), :int(feature_num * 0.8)]
    b8_label = label[int(sample_num * 0.7):int(sample_num * 0.8)]

    # 使用 PCA 将高维数据降到 2 维以便画图
    pca_b1 = PCA(n_components=2).fit_transform(b1_data)
    pca_b8 = PCA(n_components=2).fit_transform(b8_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制 Block 1
    scatter1 = ax1.scatter(pca_b1[:, 0], pca_b1[:, 1], c=b1_label, cmap='coolwarm', alpha=0.6, s=15)
    ax1.set_title(f'Block 1 (冷启动期) 数据分布\n(特征维度: {b1_data.shape[1]}维)', fontsize=13)
    ax1.set_xlabel('主成分 1 (PCA 1)')
    ax1.set_ylabel('主成分 2 (PCA 2)')

    # 绘制 Block 8
    scatter2 = ax2.scatter(pca_b8[:, 0], pca_b8[:, 1], c=b8_label, cmap='coolwarm', alpha=0.6, s=15)
    ax2.set_title(f'Block 8 (概念漂移期) 数据分布\n(特征维度: {b8_data.shape[1]}维)', fontsize=13)
    ax2.set_xlabel('主成分 1 (PCA 1)')
    ax2.set_ylabel('主成分 2 (PCA 2)')

    # 添加图例 (基于 coolwarm，通常冷色0是正常，暖色1是异常)
    handles, _ = scatter2.legend_elements(prop="colors")
    fig.legend(handles, ["正常流量 (Normal)", "异常攻击 (Anomaly)"], loc="lower center", ncol=2, fontsize=12)

    plt.suptitle('数据流前后期的特征空间分布对比 (展现概念漂移)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 请把这里的路径替换为你电脑里真实的 .npy 或 .npz 路径
    dataset_path = 'data/3_backdoor.npz'
    analyze_and_visualize(dataset_path)