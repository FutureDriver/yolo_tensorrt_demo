// ============================================================
// Copyright (c) 2026 FutureDriver
// SPDX-License-Identifier: MIT
//
// 文件：yolo_infer.cpp
// 功能：YOLOv8 TensorRT 高性能推理类实现
//       包含 GPU 预处理、端到端推理、结果解析
// 作者：FutureDriver
// 日期：2026-05-24
// ============================================================

#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "yolo_infer.hpp"

// 外部 CUDA 预处理函数（定义在 preprocess.cu 中）
// 将 BGR 8-bit 图像在 GPU 上完成 resize、颜色转换、归一化、通道重排
void launch_preprocess(
    const unsigned char* img_data,   // BGR 图像数据指针
    int width,                       // 图像宽度
    int height,                      // 图像高度
    float* gpu_dst,                  // 输出 NCHW float 缓冲区（GPU 显存）
    cudaStream_t stream              // CUDA 异步流
);

namespace {
    Logger gLogger;   // 全局 Logger，供所有 TensorRT 对象共享
}

// ----- 构造函数：加载引擎文件，反序列化，分配显存 -----
YOLOInfer::YOLOInfer(const std::string& engine_path) {
    // 1. 读取序列化引擎文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: cannot open engine file: " << engine_path << std::endl;
        return;
    }
    std::vector<char> engine_data((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());

    // 2. 创建 Runtime，反序列化引擎
    runtime_ = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>(
        nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT Runtime" << std::endl;
        return;
    }
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>(
        runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr));
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return;
    }
    context_ = std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>(
        engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return;
    }

    // 3. 创建 CUDA 异步流
    cudaStreamCreate(&stream_);

    // 4. 获取输入/输出绑定信息（端到端引擎输出形状为 [1, 300, 6]）
    int input_idx = -1, output_idx = -1;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_idx = i;
            auto dims = engine_->getTensorShape(name);
            input_w_ = dims.d[3];
            input_h_ = dims.d[2];
            input_size_ = dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3] * sizeof(float);
        } else {
            output_idx = i;
            auto dims = engine_->getTensorShape(name);
            output_size_ = dims.d[0] * dims.d[1] * dims.d[2] * sizeof(float);
        }
    }
    if (input_idx == -1 || output_idx == -1) {
        std::cerr << "Failed to find input/output bindings" << std::endl;
        return;
    }

    // 5. 分配 GPU 显存 & CPU 缓存
    void* raw_input = nullptr, *raw_output = nullptr;
    cudaMalloc(&raw_input, input_size_);
    cudaMalloc(&raw_output, output_size_);
    input_gpu_  = std::unique_ptr<void, CudaDeleter>(raw_input, CudaDeleter{});
    output_gpu_ = std::unique_ptr<void, CudaDeleter>(raw_output, CudaDeleter{});

    input_cpu_.resize(input_size_ / sizeof(float));
    output_cpu_.resize(output_size_ / sizeof(float));

    // 6. 组织 bindings 数组
    buffers_.resize(2);
    buffers_[input_idx]  = input_gpu_.get();
    buffers_[output_idx] = output_gpu_.get();

    initialized_ = true;
    std::cout << "YOLOInfer initialized successfully." << std::endl;
}

// ----- 析构函数（RAII 自动释放 GPU 资源）-----
YOLOInfer::~YOLOInfer() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

// ----- 移动构造函数 -----
YOLOInfer::YOLOInfer(YOLOInfer&& other) noexcept
    : runtime_(std::move(other.runtime_)),
      engine_(std::move(other.engine_)),
      context_(std::move(other.context_)),
      stream_(other.stream_),
      input_gpu_(std::move(other.input_gpu_)),
      output_gpu_(std::move(other.output_gpu_)),
      input_cpu_(std::move(other.input_cpu_)),
      output_cpu_(std::move(other.output_cpu_)),
      buffers_(std::move(other.buffers_)),
      input_size_(other.input_size_),
      output_size_(other.output_size_),
      input_w_(other.input_w_),
      input_h_(other.input_h_),
      initialized_(other.initialized_) {
    other.stream_ = nullptr;
    other.initialized_ = false;
}

// ----- 移动赋值运算符 -----
YOLOInfer& YOLOInfer::operator=(YOLOInfer&& other) noexcept {
    if (this != &other) {
        runtime_   = std::move(other.runtime_);
        engine_    = std::move(other.engine_);
        context_   = std::move(other.context_);
        std::swap(stream_, other.stream_);
        input_gpu_  = std::move(other.input_gpu_);
        output_gpu_ = std::move(other.output_gpu_);
        input_cpu_  = std::move(other.input_cpu_);
        output_cpu_ = std::move(other.output_cpu_);
        buffers_    = std::move(other.buffers_);
        input_size_  = other.input_size_;
        output_size_ = other.output_size_;
        input_w_     = other.input_w_;
        input_h_     = other.input_h_;
        initialized_ = other.initialized_;
        other.initialized_ = false;
        other.stream_      = nullptr;
    }
    return *this;
}

// ----- 图像预处理（GPU 加速版）-----
// 调用 preprocess.cu 中的 CUDA kernel，一步完成 resize + 颜色转换 + 归一化
void YOLOInfer::Preprocess(const cv::Mat& image) {
    launch_preprocess(
        image.data,
        image.cols, image.rows,
        static_cast<float*>(input_gpu_.get()),
        stream_
    );
}

// ----- 执行推理 -----
void YOLOInfer::DoInference() {
    context_->enqueueV2(buffers_.data(), stream_, nullptr);
}

// ----- 后处理（端到端引擎，直接解析输出 [1, 300, 6]）-----
// 每行 [x1, y1, x2, y2, confidence, class]，坐标归一化到 [0,1]
std::vector<BBox> YOLOInfer::Postprocess() {
    cudaStreamSynchronize(stream_);

    constexpr int max_dets = 300;
    std::vector<float> output_cpu(max_dets * 6);
    cudaMemcpyAsync(output_cpu.data(), output_gpu_.get(),
                    output_cpu.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<BBox> boxes;
    for (int i = 0; i < max_dets; ++i) {
        float x1   = output_cpu[i * 6 + 0];
        float y1   = output_cpu[i * 6 + 1];
        float x2   = output_cpu[i * 6 + 2];
        float y2   = output_cpu[i * 6 + 3];
        float conf = output_cpu[i * 6 + 4];
        int   cls  = static_cast<int>(output_cpu[i * 6 + 5]);

        if (conf > 0.5f) {
            boxes.push_back({x1 * input_w_, y1 * input_h_,
                             x2 * input_w_, y2 * input_h_, conf, cls});
        }
    }
    return boxes;
}

// ----- 核心推理接口 -----
// 包含耗时诊断：预热 20 次后，统计 100 次各阶段平均耗时（微秒）
std::vector<BBox> YOLOInfer::Infer(const cv::Mat& image) {
    if (!initialized_) {
        std::cerr << "Infer called but object is not initialized" << std::endl;
        return {};
    }

    static int call_count = 0;
    static std::chrono::microseconds pre_sum{0}, infer_sum{0}, post_sum{0};
    static constexpr int kDiagWarmup = 20;
    static constexpr int kDiagRuns   = 100;

    auto t0 = std::chrono::high_resolution_clock::now();
    Preprocess(image);
    auto t1 = std::chrono::high_resolution_clock::now();
    DoInference();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto boxes = Postprocess();
    auto t3 = std::chrono::high_resolution_clock::now();

    ++call_count;
    if (call_count > kDiagWarmup && call_count <= kDiagWarmup + kDiagRuns) {
        pre_sum   += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        infer_sum += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        post_sum  += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
    }
    if (call_count == kDiagWarmup + kDiagRuns) {
        std::cout << "[Profile] Preprocess avg: " << pre_sum.count() / kDiagRuns
                  << " us, Inference avg: " << infer_sum.count() / kDiagRuns
                  << " us, Postprocess avg: " << post_sum.count() / kDiagRuns
                  << " us" << std::endl;
    }

    return boxes;
}