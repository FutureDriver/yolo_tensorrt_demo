#!/usr/bin/env python3
# ============================================================
# Copyright (c) 2026 FutureDriver
# SPDX-License-Identifier: MIT
#
# 文件：plot_comparison.py
# 功能：从各优化阶段的 CSV 读取延迟，绘制 C++ 优化历程对比图
#       （对比基准为 Python PyTorch）
# 作者：FutureDriver
# 日期：2026-05-16
# ============================================================

import matplotlib
matplotlib.use('Agg')          # 容器无显示器，必须使用非交互式后端
import matplotlib.pyplot as plt
import csv
from pathlib import Path


def load_mean_ms(csv_path, framework_name):
    """
    从指定 CSV 文件中，提取某个框架的平均延迟（mean_ms）。
    
    参数：
        csv_path: CSV 文件路径
        framework_name: 要查找的框架名称（如 'TensorRT_FP16'）
    
    返回：
        浮点数 mean_ms，如果文件不存在或找不到框架则返回 None
    """
    if not Path(csv_path).exists():
        return None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)          # 自动将第一行作为列名
        for row in reader:
            name = row.get('framework', '')
            if name == framework_name:     # 找到目标框架
                try:
                    return float(row['mean_ms'])
                except (ValueError, KeyError):
                    return None
    return None


# ================================================================
# 优化阶段登记
#     每各阶段对应的测试结果文件名：
#         C++ FP16 初始版本	：  results/cpp_fp16_baseline.csv
#         GPU 预处理 CUDA 化：	results/cpp_cuda_pre.csv
#         GPU 后处理：      	results/cpp_gpu_post.csv
#         INT8 量化：           results/cpp_int8.csv
#         流水线化：        	results/cpp_pipeline.csv
#         这样一看名字就知道是哪个阶段的快照，比 cpp_benchmark.csv 清晰得多。
#     每项：(图表标签, CSV文件路径, 框架名)
#     脚本会按顺序读取，不存在的 CSV 会自动跳过，图里不会显示。
#     每完成一个优化，只需取消或增加一行即可更新图表。
# ================================================================
stages = [
    # ---------- 基准线 ----------
    (
        "PyTorch\n(Python)",                # 图表下方显示的标签
        "results/baseline_benchmark.csv",   # Python 基线 CSV
        "PyTorch"                           # CSV 中的 framework 列的值
    ),
    (
        "TensorRT FP16\n(基线)",            # C++ 第一个版本
        "results/cpp_fp16_baseline.csv",
        "TensorRT_FP16"
    ),

    # ---------- 优化阶段 1：预处理 CUDA 化 ----------
    # 生成方式：完成 GPU 预处理优化后，复制 cpp_benchmark.csv 为 cpp_benchmark_cuda_pre.csv
    (
        "TensorRT FP16\n+CUDA预处理",
        "results/cpp_cuda_pre.csv",
        "TensorRT_FP16"
    ),

    # ---------- 优化阶段 2：后处理 GPU 化 ----------
    # 生成方式：用 EfficientNMS 插件重构后，复制新的 CSV 文件
    # (
    #     "TensorRT FP16\n+GPU后处理",
    #     "results/cpp_gpu_post.csv",
    #     "TensorRT_FP16"                   # 如果框架名不变
    # ),

    # ---------- 优化阶段 3：INT8 量化 ----------
    # 生成方式：构建 INT8 引擎并测试后，复制 CSV
    # (
    #     "TensorRT INT8\n+CUDA预处理",
    #     "results/cpp_int8.csv",
    #     "TensorRT_FP16"                   # 或使用新框架名 'TensorRT_INT8'
    # ),

    # ---------- 优化阶段 4：流水线化 ----------
    # 说明：流水线化主要提升吞吐量，延迟可能不变或略增。
    #       这里先占位，后续可改为吞吐量对比图。
    # (
    #     "TensorRT INT8\n+流水线",
    #     "results/cpp_pipeline.csv",
    #     "TensorRT_FP16"
    # ),
]

# 收集实际存在的数据
labels = []
times = []

for label, csv_path, fw_name in stages:
    mean_val = load_mean_ms(csv_path, fw_name)
    if mean_val is not None:
        labels.append(label)
        times.append(mean_val)
    else:
        print(f"⚠️  跳过 {label} (文件不存在或缺少 {fw_name})")

# 如果没有任何数据，直接退出
if not times:
    print("❌ 没有任何可用数据，请检查 CSV 文件和框架名。")
    exit(1)

# ===================== 开始绘图 =====================

# 颜色方案（足以覆盖大多数优化阶段）
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, times, color=colors[:len(labels)])

# 在每个柱子上方显示具体延迟数字
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.15,
            f'{val:.2f} ms',
            ha='center', va='bottom', fontsize=11)

# 坐标轴和标题
ax.set_ylabel('Average Latency (ms)')
ax.set_title('YOLOv8n Inference Latency — C++ Optimization Journey')
ax.grid(axis='y', alpha=0.4)

# 调整纵轴范围，给标注留出空间
ymax = max(times) * 1.15
ax.set_ylim(0, ymax)

plt.tight_layout()

# 确保 results 目录存在，再保存图片
Path('results').mkdir(exist_ok=True)
plt.savefig('results/performance_chart.png', dpi=150)
print("✅ 图表已保存到 results/performance_chart.png")