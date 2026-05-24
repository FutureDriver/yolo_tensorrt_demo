#!/usr/bin/env python3
# ============================================================
# Copyright (c) 2026 FutureDriver
# SPDX-License-Identifier: MIT
#
# 文件：plot_comparison.py
# 功能：从各优化阶段的 CSV 读取延迟，绘制 C++ 优化历程对比图
#       （对比基准为 Python PyTorch），并在图上标注延迟降低百分比
# 作者：FutureDriver
# 日期：2026-05-24
# ============================================================

import os
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
            name = row.get('framework', '').strip()
            if name == framework_name:     # 找到目标框架
                try:
                    return float(row['mean_ms'])
                except (ValueError, KeyError):
                    return None
    return None


# ================================================================
#  项目根目录
#  说明：使用脚本所在位置向上推一级得到项目根目录，
#        确保无论从哪里执行脚本，CSV 路径都正确。
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."


# ================================================================
#  优化阶段登记
#     各阶段对应的测试结果文件名：
#         Python PyTorch 基线     ：  results/baseline_benchmark.csv
#         C++ FP16 初始版本	    ：  results/cpp_fp16_baseline.csv
#         GPU 预处理 CUDA 化       ：  results/cpp_cuda_pre.csv
#         端到端检测（后处理 GPU 化）： results/cpp_end2end_fp16.csv
#         INT8 量化（已编码）      ：  results/cpp_int8.csv（待环境验证）
#         流水线化（规划中）        ：  results/cpp_pipeline.csv（待实现）
#     每项：(图表标签, CSV文件相对路径, 框架名)
#     脚本会按顺序读取，不存在的 CSV 会自动跳过，图里不会显示。
#     每完成一个优化，只需取消或增加一行即可更新图表。
#     标签统一使用英文，避免容器内中文字体缺失导致乱码。
# ================================================================
stages = [
    # ---------- 基准线 ----------
    (
        "PyTorch (Baseline)",              # 英文标签，无中文字体依赖
        "results/baseline_benchmark.csv",   # Python 基线 CSV
        "PyTorch"                           # CSV 中的 framework 列的值
    ),
    (
        "TRT FP16 (Base)",                 # C++ 第一个版本
        "results/cpp_fp16_baseline.csv",
        "TensorRT_FP16"
    ),

    # ---------- 优化阶段 1：预处理 CUDA 化 ----------
    (
        "TRT FP16 + CUDA Pre",
        "results/cpp_cuda_pre.csv",
        "TensorRT_FP16"
    ),

    # ---------- 优化阶段 2：后处理 GPU 化（端到端引擎）----------
    (
        "TRT FP16 + End2End",
        "results/cpp_end2end_fp16.csv",    # 实际保存的快照文件名
        "TensorRT_FP16"
    ),

    # ---------- 优化阶段 3：INT8 量化（已编码，待环境验证）----------
    # (
    #     "TRT INT8 + CUDA Pre",
    #     "results/cpp_int8.csv",
    #     "TensorRT_FP16"
    # ),

    # ---------- 优化阶段 4：流水线化（规划中）----------
    # (
    #     "TRT INT8 + Pipeline",
    #     "results/cpp_pipeline.csv",
    #     "TensorRT_FP16"
    # ),
]

# 收集实际存在的数据
labels = []
times = []

for label, csv_rel_path, fw_name in stages:
    csv_path = os.path.join(BASE_DIR, csv_rel_path)   # 拼接出绝对路径
    mean_val = load_mean_ms(csv_path, fw_name)
    if mean_val is not None:
        labels.append(label)
        times.append(mean_val)
    else:
        print(f"⚠️  跳过 {label} (文件不存在或缺少 {fw_name})")

if not times:
    print("❌ 没有任何可用数据，请检查 CSV 文件和框架名。")
    exit(1)

# 以第一个柱（PyTorch）为基准计算延迟降低百分比
baseline = times[0] if times else None

# ===================== 开始绘图 =====================

# 颜色方案：基线用灰色，优化版本用渐变蓝绿
colors = ['#7f7f7f'] + ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd'][:len(times)-1]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(labels, times, color=colors, width=0.55)

# 在每个柱子上标注延迟（柱内白色文字，清晰易读）
for bar, val in zip(bars, times):
    height = bar.get_height()
    # 柱内延迟数值（白色加粗）
    ax.text(bar.get_x() + bar.get_width() / 2.0,
            height * 0.95,
            f'{val:.2f} ms',
            ha='center', va='top', fontsize=11, color='white', fontweight='bold')

    # 柱外上方标注：基线显示 "Baseline"，其余显示延迟降低百分比
    if baseline and baseline > 0:
        reduction = (baseline - val) / baseline * 100.0
        if abs(val - baseline) < 1e-6:          # 是基准本身
            label = "Baseline"
        else:
            label = f'-{reduction:.1f}%'
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                height + 0.15,
                label,
                ha='center', va='bottom', fontsize=10, color='black')

# 坐标轴和标题
ax.set_ylabel('Average Latency (ms)')
ax.set_title('YOLOv8n Inference Latency — C++ Optimization Journey')
ax.grid(axis='y', alpha=0.3)

# 调整纵轴范围，给标注留出空间
ymax = max(times) * 1.2
ax.set_ylim(0, ymax)

plt.tight_layout()

# 确保 results 目录存在，再保存图片
out_path = os.path.join(BASE_DIR, "results", "performance_chart.png")
plt.savefig(out_path, dpi=150)
print(f"✅ 图表已保存到 {out_path}")