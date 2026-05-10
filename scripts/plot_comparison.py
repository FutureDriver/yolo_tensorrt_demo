#!/usr/bin/env python3
# ============================================================
# Copyright (c) 2026 FutureDriver
# SPDX-License-Identifier: MIT
#
# 文件：plot_comparison.py
# 功能：从 CSV 读取性能数据，生成延迟对比柱状图
# 作者：FutureDriver
# 日期：2026-05-11
# ============================================================

import matplotlib
matplotlib.use('Agg')          # 无 GUI 后端，容器内必须
import matplotlib.pyplot as plt
import csv
from pathlib import Path

def load_csv(path):
    """读取 benchmark CSV，返回 {framework: mean_ms} 字典"""
    data = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            framework = row.get('framework', '')
            # 跳过 GPU 信息行和空行
            if 'GPU' in framework or not framework:
                continue
            try:
                data[framework] = float(row['mean_ms'])
            except (ValueError, KeyError):
                continue
    return data

# 读取两个 CSV
baseline = load_csv('results/baseline_benchmark.csv')
cpp      = load_csv('results/cpp_benchmark.csv')

# 指定期望的框架顺序
labels = ['PyTorch', 'ONNXRuntime', 'TensorRT_FP16']
try:
    times = [baseline['PyTorch'], baseline['ONNXRuntime'], cpp['TensorRT_FP16']]
except KeyError as e:
    print(f"错误：CSV 中缺少框架数据 {e}")
    exit(1)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, times, color=colors)

# 在柱上标注延迟数值
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.3,
            f'{val:.2f} ms', ha='center', fontsize=11)

ax.set_ylabel('Latencia (ms)')   # 可改为 'Latency (ms)' 看习惯
ax.set_title('YOLOv8n Inference Latency Comparison')
ax.grid(axis='y', alpha=0.5)

plt.tight_layout()
Path('results').mkdir(exist_ok=True)
plt.savefig('results/performance_chart.png', dpi=150)
print("图表已保存到 results/performance_chart.png")