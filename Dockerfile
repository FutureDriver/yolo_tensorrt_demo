# ============================================================
# Copyright (c) 2026 FutureDriver
# SPDX-License-Identifier: MIT
#
# 文件：Dockerfile
# 功能：构建 YOLOv8n TensorRT 推理 Demo 的运行环境
# 作者：FutureDriver
# 日期：2026-05-26
# ============================================================

#FROM nvcr.io/nvidia/tensorrt:23.08-py3

# 安装系统依赖
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#        libopencv-dev \
#        libgl1-mesa-glx \
#        libglib2.0-0 \
#        wget \
#    && rm -rf /var/lib/apt/lists/*

# 使用清华源加速 Python 包安装
# 必须同时安装 onnxruntime-gpu（GPU 推理）和 onnxruntime（CPU 版，供 ultralytics 校验）
#RUN pip install --no-cache-dir \
#        -i https://pypi.tuna.tsinghua.edu.cn/simple \
#        ultralytics \
#        onnxruntime-gpu \
#        onnxruntime \
#        onnx \
#        onnxslim \
#        matplotlib \
#        pandas \
#        opencv-python-headless

#WORKDIR /workspace/demo
#COPY . .

#CMD ["/bin/bash"]

#FROM trt_demo_env:latest

#WORKDIR /workspace/demo
#COPY . .

#CMD ["/bin/bash"]

# GitHub Container Registry 
# (GitHub 容器镜像仓库（也常译为 GitHub 容器注册表 或 GitHub 容器仓库）)
#FROM ghcr.io/futuredriver/yolo-trt-demo-base:latest

#WORKDIR /workspace/demo
#COPY . .

#CMD ["/bin/bash"]

# 阿里云
FROM crpi-ztrpo5sgrnerrgv0.cn-hangzhou.personal.cr.aliyuncs.com/futuredriver/yolo-trt-demo:latest

WORKDIR /workspace/demo
COPY . .

CMD ["/bin/bash"]