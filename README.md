<!-- ============================================================
Copyright (c) 2026 FutureDriver
SPDX-License-Identifier: MIT

文件：README.md
功能：项目说明文档
作者：FutureDriver
日期：2026-05-17
============================================================ -->

# YOLOv8 TensorRT C++ 高性能部署 Demo

> 从 ONNX 导出到 C++ TensorRT 推理的完整工程实践，展示 RAII 资源管理、异步 CUDA 流、移动语义等现代 C++ 最佳实践。

## 📊 性能概览

| 框架 | 平均延迟 | P95 延迟 | 吞吐量 (FPS) | 相对 PyTorch 提升 |
|------|----------|----------|--------------|-------------------|
| Python PyTorch (Baseline) | 7.59 ms | 8.51 ms | 131.7 | — |
| C++ TensorRT FP16 (初始) | 6.22 ms | 6.94 ms | 160.9 | -18.0% |
| **C++ TensorRT FP16 + CUDA 预处理** | **4.50 ms** | **5.24 ms** | **222.0** | **-40.7%** |

![性能对比图](results/performance_chart.png)

> **阶段耗时分解**（当前最优版本）：  
> - 预处理 (GPU)：1323 us (29%)  
> - 推理 (GPU)：994 us (22%)  
> - 后处理 (CPU)：2319 us (52%)  
>
> 推理核心已压至 **0.99 ms**，比 Python 侧推理运算快 5 倍以上。当前瓶颈在后处理（CPU 端 NMS），下一步将后处理迁移至 GPU，预期总延迟降至 2.5 ms 以内。

## 🚀 快速开始

### 环境要求
- NVIDIA GPU（GTX 1060 6GB+，推荐 RTX 3060+）
- Docker + NVIDIA Container Toolkit
- 约 15 GB 磁盘空间

### 编译与运行
```bash
# 1. 拉取 TensorRT 镜像（已完成则跳过）
docker pull nvcr.io/nvidia/tensorrt:23.08-py3

# 2. 编译项目（支持 CUDA，需使用带 CUDA 的编译镜像）
docker run --rm --gpus all -v $(pwd):/workspace/demo trt_demo_env \
  bash -c "cd /workspace/demo/build && cmake .. && make -j"

# 3. 运行 Python 基线（生成 baseline_benchmark.csv）
docker run --rm --gpus all -w /workspace/demo -v $(pwd):/workspace/demo trt_demo_env \
  python3 scripts/baseline_benchmark.py

# 4. 运行 C++ TensorRT 基准测试（输出 cpp_benchmark.csv）
docker run --rm --gpus all -w /workspace/demo -v $(pwd):/workspace/demo trt_demo_env \
  ./build/benchmark

# 5. 生成性能对比图
docker run --rm --gpus all -w /workspace/demo -v $(pwd):/workspace/demo trt_demo_env \
  python3 scripts/plot_comparison.py
```

## 🔧 技术要点
- **RAII 资源管理**：所有 CUDA 资源（Runtime、Engine、Context、CUDA Stream、GPU 显存）均用 `std::unique_ptr` + 自定义 Deleter 封装，确保异常安全，代码中零裸指针。
- **移动语义**：推理类禁止拷贝，实现 `noexcept` 移动构造函数/赋值运算符，保证高性能数据流转。
- **异步 CUDA 流**：预处理、推理、后处理在独立 CUDA Stream 上异步执行，通过 `cudaStreamSynchronize` 精确控制同步点。
- **GPU 预处理**：使用 NPP 库进行图像 resize，自定义 CUDA kernel 完成 BGR→RGB、归一化、HWC→CHW 转换，全程 GPU 执行，预处理耗时仅 1.3 ms。
- **FP16 量化**：TensorRT 构建时开启 FP16 模式，推理核心延迟仅 **0.99 ms**。
- **编译期优化**：用 `constexpr` 定义常量、`std::string_view` 避免不必要的内存分配，启用 C++17 标准。

## 📁 项目结构
```
.
├── CMakeLists.txt          # CMake 构建配置（含 CUDA 支持）
├── LICENSE
├── README.md
├── include/
│   └── yolo_infer.hpp      # 推理类声明
├── src/
│   ├── build_engine.cpp    # TensorRT 引擎构建工具
│   ├── yolo_infer.cpp      # 推理类实现
│   ├── preprocess.cu       # GPU 预处理（NPP resize + CUDA kernel）
│   ├── main.cpp            # 快速演示程序
│   └── benchmark.cpp       # 性能基准测试（自动对比 Python 基线）
├── scripts/
│   ├── baseline_benchmark.py  # Python 基线测试
│   ├── export_onnx.py         # 导出 ONNX（可选）
│   ├── verify_onnx.py         # ONNX 模型验证
│   └── plot_comparison.py     # 性能对比绘图（柱状图 + 降低百分比）
├── data/
│   └── demo.jpg            # 测试图片
├── models/                 # 模型文件（.pt / .onnx / .engine，不提交 Git）
├── results/                # 测试结果 CSV 及性能图
└── build/                  # 编译中间文件（不提交 Git）
```

## 📝 优化路线图
- [✓] **预处理 CUDA 化**：使用 NPP resize + CUDA kernel 完成缩放与颜色转换，预处理延迟从 2.6ms 降至 1.3ms。
- [ ] **后处理 GPU 化**：将 CPU 端 NMS 替换为 TensorRT EfficientNMS 插件，后处理预计从 2.3ms 降至 0.2ms 以内。
- [~] **INT8 量化**：校准器与构建逻辑已完成，因 WSL2 Docker 环境限制未完成端到端测试，计划在原生 Linux / Jetson 上验证
- [ ] **流水线化**：多线程分离 I/O 与推理，提升连续帧吞吐量至 400+ FPS。

## 📄 许可证
MIT License
```
