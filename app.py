import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 导入你原有的模块
from load_data import *
from graph_generation import graph_generation, graph_incremental
from hypersphere import HyperSphere_Construction
from metric import mask_list
from parameter import *

# ================= 1. 大屏全局配置 =================
st.set_page_config(
    page_title="OODOFS + METER 实时流式异常检测系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 面向开放特征空间的在线异常检测实时监控")
st.markdown("""
**创新点融合**：基于 OODOFS 基线，融入 $\ell_{1,2}$-范数稀疏化特征淘汰（**SAD**）与动态概念漂移自适应（**METER**）。
""")

# ================= 2. 侧边栏：控制面板 =================
with st.sidebar:
    st.header("⚙️ 算法控制面板")
    run_button = st.button("▶️ 启动实时数据流", use_container_width=True)
    stop_button = st.button("⏹️ 停止运行", use_container_width=True)

    st.divider()
    st.subheader("超参数设置")
    lambda_sparse = st.slider("特征稀疏惩罚系数 (λ)", min_value=0.0001, max_value=0.01, value=0.0001, step=0.0001,
                              format="%.4f")
    sleep_time = st.slider("数据流刷新延迟 (秒/块)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# ================= 3. 布局占位符初始化 =================
# 顶部：核心指标 KPI
kpi_cols = st.columns(4)
kpi_acc = kpi_cols[0].empty()
kpi_f1 = kpi_cols[1].empty()
kpi_drift = kpi_cols[2].empty()
kpi_sparsity = kpi_cols[3].empty()

# 中部：动态图表
chart_cols = st.columns(2)
drift_chart_placeholder = chart_cols[0].empty()
sparsity_chart_placeholder = chart_cols[1].empty()

# 底部：实时报警日志
st.subheader("🚨 智能演化控制器 (IEC) 实时报警日志")
log_placeholder = st.empty()

# ================= 4. 核心流式运行逻辑 =================
if run_button:
    # 模拟日志记录
    logs = []

    # 准备数据
    data_path = "data/3_backdoor.npz"
    data_list, label_list = tkde_newdataset_normalization(data_path, random_seed=1020)

    model_layers = 2
    training_epoch = 100
    reduced_size = return_reduced_size('backdoor')
    label_rate = 0.2

    # 初始化模型
    truth_label = []
    Starter = HyperSphere_Construction(reduced_size=reduced_size, layer_num=model_layers, Epoch=training_epoch)

    # 记录动态画图的数据
    plot_drifts = []
    plot_thresholds = []
    plot_sparsity = []
    plot_f1s = []

    progress_bar = st.progress(0)

    # 遍历数据流 (Data Stream)
    for times, data in enumerate(data_list):
        if stop_button:
            break

        progress_bar.progress((times + 1) / len(data_list))

        # ----------------- 步骤A：模型运行 -----------------
        sample_num, feature_num = data.size()
        label = label_list[times]
        mask_data = mask_list(data, label_rate)

        if times == 0:
            graph_data = graph_generation(data, reduced_size, label)
            truth_label.extend(graph_data.y.numpy())
            # 修改你的源代码，将 lambda_sparse 传入（如果不方便修改，源代码固定 0.0001 也行）
            Starter.First_Construction(graph_data=graph_data, mask=mask_data, feature_num=feature_num, data=data)
            current_drift = 0.0
            is_drift = False
        else:
            old_sample_num, old_feature_num = sample_num, feature_num
            graph_data = graph_incremental(graph_data.x[:old_feature_num].detach().cpu(), data, label,
                                           feature_num - old_feature_num, reduced_size)
            truth_label.extend(graph_data.y.numpy())
            Starter.Second_Construction(graph_data=graph_data, mask=mask_data, feature_num=feature_num,
                                        old_feature_num=old_feature_num)

            # 获取最新状态
            current_drift = Starter.drift_history[-1]
            # 计算当前动态阈值
            if len(Starter.drift_history) >= 6:
                window = Starter.drift_history[-6:-1]
                threshold = np.mean(window) + 2 * np.std(window)
            else:
                threshold = np.nan
            plot_thresholds.append(threshold)

            # 判定是否漂移用于大屏报警
            is_drift = (current_drift > threshold and current_drift > 0.05) if not np.isnan(threshold) else False

        # 收集画图数据
        plot_drifts.append(current_drift if times > 0 else 0)
        current_sparsity = Starter.sparsity_history[-1] if hasattr(Starter, 'sparsity_history') and len(
            Starter.sparsity_history) > 0 else 0
        plot_sparsity.append(current_sparsity)

        # 计算实时评价指标
        current_acc = accuracy_score(y_true=np.array(truth_label), y_pred=np.array(Starter.predictionlist))
        current_f1 = f1_score(y_true=np.array(truth_label), y_pred=np.array(Starter.predictionlist), pos_label=0)

        # ----------------- 步骤B：大屏 UI 实时更新 -----------------
        # 1. 更新顶部 KPI
        kpi_acc.metric("实时准确率 (Accuracy)", f"{current_acc:.2%}")
        kpi_f1.metric("实时 F1-Score", f"{current_f1:.4f}")

        if times == 0:
            plot_thresholds.append(np.nan)

        drift_delta = current_drift - (plot_drifts[-2] if len(plot_drifts) > 1 else 0)
        kpi_drift.metric("当前概念漂移得分", f"{current_drift:.4f}", delta=f"{drift_delta:.4f}", delta_color="inverse")

        sparsity_delta = current_sparsity - (plot_sparsity[-2] if len(plot_sparsity) > 1 else 0)
        kpi_sparsity.metric("实时特征稀疏率", f"{current_sparsity:.2%}", delta=f"{sparsity_delta:.2%}")

        # 2. 如果发生漂移，触发报警与日志
        if is_drift:
            st.toast(f"⚠️ 检测到概念漂移！得分: {current_drift:.2f}", icon="🚨")
            logs.insert(0,
                        f"🔴 [Block {times}] 警报触发！概念不确定性激增至 {current_drift:.4f}，触发动态自适应 (METER) 🚀")
        else:
            logs.insert(0, f"🟢 [Block {times}] 数据流平稳。特征矩阵稀疏化进行中... 稀疏率: {current_sparsity:.2%}")

        # 仅保留最近 6 条日志显示
        log_html = "<div style='height: 150px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>"
        for log in logs[:6]:
            log_html += f"<p style='font-family: monospace; margin: 2px;'>{log}</p>"
        log_html += "</div>"
        log_placeholder.markdown(log_html, unsafe_allow_html=True)

        # 3. 动态渲染 matplotlib 图表
        # 图1：概念漂移自适应图
        fig_drift, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(plot_drifts, label='Drift Score', color='royalblue', marker='.', markersize=4)
        ax1.plot(plot_thresholds, label='Dynamic Threshold', color='darkorange', linestyle='--')
        # 画红点
        for i in range(len(plot_drifts)):
            if not np.isnan(plot_thresholds[i]) and plot_drifts[i] > plot_thresholds[i] and plot_drifts[i] > 0.05:
                ax1.scatter(i, plot_drifts[i], color='crimson', s=100, zorder=5)
        ax1.set_title("Concept Drift Monitoring (METER)", fontsize=10)
        ax1.legend(loc='upper right', fontsize=8)
        drift_chart_placeholder.pyplot(fig_drift)
        plt.close(fig_drift)

        # 图2：特征稀疏率图
        fig_spar, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(plot_sparsity, color='seagreen', marker='s', markersize=4)
        ax2.fill_between(range(len(plot_sparsity)), plot_sparsity, color='seagreen', alpha=0.2)
        import matplotlib.ticker as ticker

        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax2.set_title("Feature Sparsity Ratio ($\ell_{1,2}$-norm)", fontsize=10)
        sparsity_chart_placeholder.pyplot(fig_spar)
        plt.close(fig_spar)

        # 模拟流式注入延迟
        time.sleep(sleep_time)

    st.success(f"🎉 数据流处理完毕！最终 F1-Score: {current_f1:.4f}")
    st.balloons()

#python -m streamlit run app.py