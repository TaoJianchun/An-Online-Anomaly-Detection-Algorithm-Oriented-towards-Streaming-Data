import numpy as np

# 加载你的数据集
data_path = "data/3_backdoor.npz"
mat = np.load(data_path)
label = mat['y']

# 统计标签分布
count_0 = np.sum(label == 0)
count_1 = np.sum(label == 1)

print(f"=== 数据集标签分布探勘 ===")
print(f"原始标签 0 的样本数: {count_0}")
print(f"原始标签 1 的样本数: {count_1}")

if count_0 > count_1:
    print(f"\n结论：标签 1 是少数类（占比 {count_1/len(label)*100:.2f}%），所以【1 代表异常（后门攻击）】。")
else:
    print(f"\n结论：标签 0 是少数类（占比 {count_0/len(label)*100:.2f}%），所以【0 代表异常（后门攻击）】。")