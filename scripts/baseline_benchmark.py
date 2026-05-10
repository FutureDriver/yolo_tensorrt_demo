#!/usr/bin/env python3
# ============================================================
# Copyright (c) 2026 FutureDriver
# SPDX-License-Identifier: MIT
#
# 文件：baseline_benchmark.py
# 功能：Python 推理基线测试（PyTorch vs ONNX Runtime）
#       采用统一的预处理逻辑，消除测试偏差
# 作者：FutureDriver
# 日期：2026-05-10
# ============================================================

import time
import csv
import subprocess
import numpy as np
import cv2
import torch
import onnxruntime as ort

# ---------- 配置 ----------
PT_MODEL_PATH   = "models/yolov8n.pt"
ONNX_MODEL_PATH = "models/yolov8n.onnx"
IMAGE_PATH      = "data/demo.jpg"
OUTPUT_CSV      = "results/baseline_benchmark.csv"

NUM_WARMUP = 50   # 预热次数
NUM_RUNS   = 100  # 正式测试次数
INPUT_SIZE = (640, 640)

# ---------- 1. 准备统一输入 ----------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"测试图片不存在: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
# 归一化 + HWC -> NCHW
input_blob = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0
input_blob = np.ascontiguousarray(input_blob)

# ---------- 2. 加载模型 ----------
import ultralytics
model_pt = ultralytics.YOLO(PT_MODEL_PATH).model
model_pt.cuda()   # 把模型权重也移到 GPU
model_pt.eval()

sess_ort = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])
input_name = sess_ort.get_inputs()[0].name

# ---------- 3. 预热（同时完成显存分配）----------
print(f"预热 {NUM_WARMUP} 次 ...")
for _ in range(NUM_WARMUP):
    with torch.no_grad():
        _ = model_pt(torch.from_numpy(input_blob).cuda())
    _ = sess_ort.run(None, {input_name: input_blob})

# ---------- 4. 正式测试 ----------
print(f"正式测试 {NUM_RUNS} 次 ...")
pt_times, ort_times = [], []

for i in range(NUM_RUNS):
    # PyTorch
    start = time.perf_counter()
    with torch.no_grad():
        _ = model_pt(torch.from_numpy(input_blob).cuda())
    torch.cuda.synchronize()   # 等待 GPU 操作完成，保证测量完整
    pt_times.append((time.perf_counter() - start) * 1000)

    # ONNX Runtime
    start = time.perf_counter()
    _ = sess_ort.run(None, {input_name: input_blob})
    ort_times.append((time.perf_counter() - start) * 1000)

# ---------- 5. 统计 ----------
def stats(arr):
    arr = np.array(arr)
    return {
        "mean": np.mean(arr),
        "std":  np.std(arr),
        "min":  np.min(arr),
        "max":  np.max(arr),
        "p95":  np.percentile(arr, 95),
        "p99":  np.percentile(arr, 99),
        "fps":  1000.0 / np.mean(arr)   # 吞吐量 = 1000ms / 平均延迟
    }

pt_stats  = stats(pt_times)
ort_stats = stats(ort_times)

# ---------- 6. 显存占用 ----------
try:
    mem_used = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
    ).decode().strip()
except:
    mem_used = "N/A"

# ---------- 7. 打印结果 ----------
print(f"\n{'='*60}")
print(f"PyTorch     : mean={pt_stats['mean']:.2f}ms, std={pt_stats['std']:.2f}ms, "
      f"p95={pt_stats['p95']:.2f}ms, FPS={pt_stats['fps']:.1f}")
print(f"ONNX Runtime: mean={ort_stats['mean']:.2f}ms, std={ort_stats['std']:.2f}ms, "
      f"p95={ort_stats['p95']:.2f}ms, FPS={ort_stats['fps']:.1f}")
print(f"GPU memory used: {mem_used} MiB")
print(f"结果已保存到 {OUTPUT_CSV}")

# ---------- 8. 保存 CSV ----------
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['framework', 'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'p95_ms', 'p99_ms', 'fps'])
    writer.writerow(['PyTorch'] + [pt_stats[k] for k in ['mean','std','min','max','p95','p99','fps']])
    writer.writerow(['ONNXRuntime'] + [ort_stats[k] for k in ['mean','std','min','max','p95','p99','fps']])
    writer.writerow([])
    writer.writerow(['GPU memory used', f'{mem_used} MiB'])