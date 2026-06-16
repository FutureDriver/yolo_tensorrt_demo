#!/bin/bash
# ============================================================
# Copyright (c) 2026 FutureDriver
# SPDX-License-Identifier: MIT
#
# 文件：run_demo.sh
# 功能：一键构建环境、编译、运行基准测试并生成性能图
# 作者：FutureDriver
# 日期：2026-05-26
# ============================================================

set -e

echo "=========================================="
echo " YOLOv8 TensorRT Demo - One-click Runner"
echo "=========================================="

# 1. 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker。"
    exit 1
fi

# 2. 检查 GPU 支持
if ! docker run --rm --gpus all yolo-trt-demo nvidia-smi &> /dev/null; then
    echo "❌ Docker 无法访问 GPU，请重新配置 nvidia-ctk runtime。"
    exit 1
fi

echo "✅ 环境检查通过。"

# 3. 构建镜像（首次约 5-10 分钟）
if ! docker image inspect yolo-trt-demo &> /dev/null; then
    echo ""
    echo "🔨 步骤 1/5：构建 Docker 镜像（首次运行需要几分钟）..."
    docker build -t yolo-trt-demo .
    echo "✅ 镜像构建完成。"
else
    echo "📦 步骤 1/5：镜像已存在，跳过构建。"
fi

# 4. 检查模型文件
if [ ! -f models/yolov8n.pt ]; then
    echo "❌ 模型文件 models/yolov8n.pt 不存在！请确保已从 GitHub 克隆完整仓库。"
    exit 1
fi

# 5. 在容器内完成全部准备和编译
echo ""
echo "🚀 步骤 2/5：启动容器，开始模型准备与编译..."
docker run --rm --gpus all \
    -v "$(pwd):/workspace/demo" \
    -w /workspace/demo \
    yolo-trt-demo \
    bash -c "
set -e

# 导出 ONNX
if [ ! -f models/yolov8n.onnx ]; then
    echo '  [2.1] 导出带 EfficientNMS 的 ONNX ...'
    python3 -c \"
from ultralytics import YOLO
model = YOLO('models/yolov8n.pt')
model.export(format='onnx', nms=True, opset=12, imgsz=640, simplify=True)
\"
    echo '  完成。'
else
    echo '  [2.1] ONNX 文件已存在，跳过。'
fi

# 编译项目
echo '  [2.2] 编译 C++ 项目 ...'
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j\$(nproc)
cd ..
echo '  完成。'

# 构建 TensorRT 引擎
if [ ! -f models/yolov8n_fp16.engine ]; then
    echo '  [2.3] 构建 TensorRT 引擎（约 1-2 分钟）...'
    ./build/build_engine
    echo '  完成。'
else
    echo '  [2.3] 引擎文件已存在，跳过。'
fi
"

# 6. 运行 Python 基线测试
echo ""
echo "📊 步骤 3/5：运行 Python 基线测试..."
docker run --rm --gpus all \
    -v "$(pwd):/workspace/demo" \
    -w /workspace/demo \
    yolo-trt-demo \
    python3 scripts/baseline_benchmark.py

# 7. 运行 C++ benchmark 并重命名为阶段快照
echo ""
echo "📊 步骤 4/5：运行 C++ TensorRT 基准测试..."
docker run --rm --gpus all \
    -v "$(pwd):/workspace/demo" \
    -w /workspace/demo \
    yolo-trt-demo \
    ./build/benchmark
mv results/cpp_benchmark.csv results/cpp_end2end_fp16.csv

# 8. 生成性能对比图
echo ""
echo "📈 步骤 5/5：生成性能对比图..."
docker run --rm --gpus all \
    -v "$(pwd):/workspace/demo" \
    -w /workspace/demo \
    yolo-trt-demo \
    python3 scripts/plot_comparison.py

echo ""
echo "=========================================="
echo " 🎉 全部完成！"
echo " 查看结果："
echo "   - Python 基线：results/baseline_benchmark.csv"
echo "   - C++ 结果：results/cpp_end2end_fp16.csv"
echo "   - 性能图表：results/performance_chart.png"
echo "=========================================="