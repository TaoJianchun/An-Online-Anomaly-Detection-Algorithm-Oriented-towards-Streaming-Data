import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from load_data import tkde_newdataset_normalization
import warnings

warnings.filterwarnings('ignore')


def run_baselines(data_path):
    print("加载数据中...")
    data_list, label_list = tkde_newdataset_normalization(data_path, random_seed=1020)

    # 初始化三个对比模型
    models = {
        "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.0244, random_state=42),
        "One-Class SVM": OneClassSVM(nu=0.0244, kernel="rbf", gamma="scale"),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.0244, novelty=True)
    }

    results = {name: {"truth": [], "pred": [], "score": []} for name in models.keys()}

    for times, (data, label) in enumerate(zip(data_list, label_list)):
        # 将张量转为 numpy 供 sklearn 使用
        X = data.numpy()
        # 将原始标签中 == 1 的视为 1（正常）， == -1 的视为 0（异常）
        # 这是为了与你模型中的标签逻辑 (0代表真实攻击) 保持绝对一致！
        y_true = np.where(label.numpy() == 1, 1, 0)

        print(f"处理数据块 {times + 1}/10, 样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

        for name, model in models.items():
            if times == 0:
                # 第一块数据：冷启动训练，并对当前块进行预测
                model.fit(X)

                # LOF 的 novelty mode 预测需要特殊处理
                y_pred = model.predict(X)
                scores = model.decision_function(X) if name != "Local Outlier Factor" else model.score_samples(X)

            else:
                # 应对开放特征空间(OFS)：传统模型无法处理特征增多
                # 只能使用上一轮模型训练时见过的特征维度 (old_feature_num) 进行【预测】
                old_feature_num = models_feature_dims[name]
                X_truncated = X[:, :old_feature_num]

                y_pred = model.predict(X_truncated)
                scores = model.decision_function(
                    X_truncated) if name != "Local Outlier Factor" else model.score_samples(X_truncated)

                # 预测完后，使用全新的、包含新特征的数据进行【重训练】（模拟最强的离线批处理基线）
                model.fit(X)

            # Sklearn 异常检测输出：1 代表正常，-1 代表异常
            # 我们将其转换为：1 代表正常，0 代表异常（与你的代码完美对齐）
            y_pred_mapped = np.where(y_pred == 1, 1, 0)

            results[name]["truth"].extend(y_true)
            results[name]["pred"].extend(y_pred_mapped)

            # 分数取反，使得异常点(0类)的得分更高，符合 AUC 计算逻辑
            results[name]["score"].extend(-scores)

            # 记录本轮训练的特征数，供下一轮截断使用
        models_feature_dims = {name: X.shape[1] for name in models.keys()}

    # 打印最终对比结果
    print("\n" + "=" * 50)
    print("🏆 传统经典算法基线对比结果 🏆")
    print("=" * 50)

    for name in models.keys():
        truth = np.array(results[name]["truth"])
        pred = np.array(results[name]["pred"])
        score = np.array(results[name]["score"])

        acc = accuracy_score(y_true=truth, y_pred=pred)
        # 注意：pos_label=0，严格评测对异常攻击的捕捉能力！
        f1 = f1_score(y_true=truth, y_pred=pred, pos_label=0)
        rec = recall_score(y_true=truth, y_pred=pred, pos_label=0)
        pre = precision_score(y_true=truth, y_pred=pred, pos_label=0)
        auc = roc_auc_score(y_true=(1 - truth), y_score=score)  # 1-truth 使异常标签变为1算AUC

        print(f"🔹 {name}:")
        print(f"   Acc ={acc:.4f}, F1 ={f1:.4f}, Recall={rec:.4f}, Precision={pre:.4f}, AUC-ROC={auc:.4f}\n")


if __name__ == '__main__':
    run_baselines("data/3_backdoor.npz")