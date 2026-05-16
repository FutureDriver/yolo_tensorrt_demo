#!/usr/bin/env python3
# ============================================================
# Copyright (c) 2026 FutureDriver
# SPDX-License-Identifier: MIT
#
# 文件：verify_onnx.py
# 功能：验证导出的 ONNX 模型结构合法，且与 PyTorch 原模型推理结果一致
# 作者：FutureDriver
# 日期：2026-05-10
# ============================================================

import onnx
import onnxruntime as ort
import numpy as np
import cv2
import torch
from ultralytics import YOLO

def main():
    onnx_path = "models/yolov8n.onnx"
    pt_path   = "models/yolov8n.pt"
    img_path  = "data/demo.jpg"

    # ---------- 1. 静态结构检查 ----------
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX 模型结构合法")

    # ---------- 2. 准备真实图像输入 ----------
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"测试图片不存在: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (640, 640))
    # 归一化 + NCHW
    input_data = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0

    # ---------- 3. PyTorch 原始输出 ----------
    model = YOLO(pt_path).model   # 获取内部 PyTorch nn.Module
    model.eval()
    with torch.no_grad():
        torch_out = model(torch.from_numpy(input_data))
        if isinstance(torch_out, (list, tuple)):
            torch_out = torch_out[0]
        torch_out = torch_out.cpu().numpy()

    # ---------- 4. ONNX Runtime 原始输出 ----------
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: input_data})[0]

    # ---------- 5. 数值比对 ----------
    if torch_out.shape != onnx_out.shape:
        print(f"❌ 输出形状不一致: PyTorch {torch_out.shape} vs ONNX {onnx_out.shape}")
        return
    diff = np.abs(torch_out - onnx_out)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f}")
    if max_diff < 1e-4:
        print("✅ ONNX 导出数值完全一致（浮点误差可忽略）")
    else:
        print("❌ 存在较大差异，请检查导出参数或算子版本")

if __name__ == "__main__":
    main()
