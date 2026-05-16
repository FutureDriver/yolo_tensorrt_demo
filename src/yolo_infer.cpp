// ============================================================
// Copyright (c) 2026 FutureDriver
// SPDX-License-Identifier: MIT
//
// 文件：yolo_infer.cpp
// 功能：YOLOv8 TensorRT 高性能推理类实现
// 作者：FutureDriver
// 日期：2026-05-06
// ============================================================

#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <opencv2/dnn.hpp>      // NMS
#include "yolo_infer.hpp"

// ----- 外部 CUDA 预处理函数（定义在 preprocess.cu 中） -----
void launch_preprocess(
    const unsigned char* img_data,   // BGR 图像数据指针
    int width,                       // 图像宽度
    int height,                      // 图像高度
    float* gpu_dst,                  // 输出 NCHW float 缓冲区（GPU）
    cudaStream_t stream              // CUDA 流
);

namespace {
    // 全局 Logger 实例，供所有 TensorRT 对象共享
    Logger gLogger;
}

// ----- 构造函数 -----
YOLOInfer::YOLOInfer(const std::string& engine_path) {
    // 1. 读取序列化引擎文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: cannot open engine file: " << engine_path << std::endl;
        return;
    }
    std::vector<char> engine_data((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());

    // 2. 创建 TensorRT Runtime，反序列化引擎
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

    // 4. 获取输入 / 输出绑定信息（动态获取尺寸，无需硬编码）
    int input_idx = -1, output_idx = -1;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_idx = i;
            auto dims = engine_->getTensorShape(name);
            input_w_ = dims.d[3];   // NCHW 格式
            input_h_ = dims.d[2];
            input_size_ = dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3] * sizeof(float);
        } else {
            output_idx = i;
            auto dims = engine_->getTensorShape(name);
            output_num_classes_ = dims.d[1] - 4;   // 前4位为 bbox 坐标
            output_num_anchors_ = dims.d[2];
            output_size_ = dims.d[0] * dims.d[1] * dims.d[2] * sizeof(float);
        }
    }
    if (input_idx == -1 || output_idx == -1) {
        std::cerr << "Failed to find input/output bindings" << std::endl;
        return;
    }

    // 5. 分配 GPU 显存 & CPU 缓存
    // 输入输出 GPU 缓冲区
    void* raw_input = nullptr, *raw_output = nullptr;
    cudaMalloc(&raw_input, input_size_);
    cudaMalloc(&raw_output, output_size_);
    input_gpu_ = std::unique_ptr<void, CudaDeleter>(raw_input, CudaDeleter{});
    output_gpu_ = std::unique_ptr<void, CudaDeleter>(raw_output, CudaDeleter{});

    // CPU 端 buffer
    input_cpu_.resize(input_size_ / sizeof(float));
    output_cpu_.resize(output_size_ / sizeof(float));

    // 6. 组织 bindings 数组（按输入输出顺序）
    buffers_.resize(2);
    buffers_[input_idx] = input_gpu_.get();
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
    // unique_ptr 会自动调用 TrtDeleter / CudaDeleter 释放资源
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
      output_num_anchors_(other.output_num_anchors_),
      output_num_classes_(other.output_num_classes_),
      initialized_(other.initialized_) {
    // 将源对象置于可安全析构的状态
    other.stream_ = nullptr;
    other.initialized_ = false;
}

// ----- 移动赋值运算符 -----
YOLOInfer& YOLOInfer::operator=(YOLOInfer&& other) noexcept {
    if (this != &other) {
        // 先释放当前资源（析构函数不会自动调用，此处手动清理？借助 swap）
        // 实际可重用析构逻辑，这里使用 swap 惯用法
        runtime_ = std::move(other.runtime_);
        engine_ = std::move(other.engine_);
        context_ = std::move(other.context_);
        std::swap(stream_, other.stream_);
        input_gpu_ = std::move(other.input_gpu_);
        output_gpu_ = std::move(other.output_gpu_);
        input_cpu_ = std::move(other.input_cpu_);
        output_cpu_ = std::move(other.output_cpu_);
        buffers_ = std::move(other.buffers_);
        input_size_ = other.input_size_;
        output_size_ = other.output_size_;
        input_w_ = other.input_w_;
        input_h_ = other.input_h_;
        output_num_anchors_ = other.output_num_anchors_;
        output_num_classes_ = other.output_num_classes_;
        initialized_ = other.initialized_;
        other.initialized_ = false;
        other.stream_ = nullptr;
    }
    return *this;
}

// ----- 图像预处理（GPU 加速版）-----
// 将 BGR8 图像直接转换为 NCHW 格式的 float 张量，并存放在 GPU 显存中
void YOLOInfer::Preprocess(const cv::Mat& image) {
    launch_preprocess(
        image.data,                              // 原始像素数据（BGR 8‑bit）
        image.cols, image.rows,                  // 图像宽高
        static_cast<float*>(input_gpu_.get()),   // 输出到 GPU 输入缓冲区NCHW
        stream_                                  // 异步流
    );
    // 预处理结果已在 GPU 上，无需额外的内存拷贝
}

// ----- 执行推理 -----
void YOLOInfer::DoInference() {
    // enqueueV2 异步执行，传入 CUDA 流
    context_->enqueueV2(buffers_.data(), stream_, nullptr);
}

// ----- 后处理与 NMS -----
std::vector<BBox> YOLOInfer::Postprocess() {
    // 等待所有异步操作完成
    cudaStreamSynchronize(stream_);

    // 将输出从 GPU 拷贝回 CPU
    cudaMemcpyAsync(output_cpu_.data(), output_gpu_.get(), output_size_,
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    const int num_classes = output_num_classes_;
    const int num_anchors = output_num_anchors_;
    const float* out = output_cpu_.data();

    std::vector<BBox> proposals;
    std::vector<float> confidences;
    std::vector<int> labels;

    // 遍历所有锚点
    for (int i = 0; i < num_anchors; ++i) {
        const float* row = out + i * (4 + num_classes);
        float x_center = row[0];
        float y_center = row[1];
        float w = row[2];
        float h = row[3];

        // 找到最大置信度及对应类别
        float max_conf = 0.f;
        int best_label = -1;
        for (int c = 0; c < num_classes; ++c) {
            float conf = row[4 + c];
            if (conf > max_conf) {
                max_conf = conf;
                best_label = c;
            }
        }

        if (max_conf > 0.5f) {
            // 将中心点 / 宽高转换为左上角和右下角（坐标已经归一化到 0~1，乘以输入尺寸）
            float x1 = (x_center - w / 2.0f) * input_w_;
            float y1 = (y_center - h / 2.0f) * input_h_;
            float x2 = (x_center + w / 2.0f) * input_w_;
            float y2 = (y_center + h / 2.0f) * input_h_;

            proposals.push_back({x1, y1, x2, y2, max_conf, best_label});
            confidences.push_back(max_conf);
            labels.push_back(best_label);
        }
    }

    // 转换为 OpenCV 需要的矩形格式 (x, y, width, height)
    std::vector<cv::Rect2d> rects;
    rects.reserve(proposals.size());
    for (const auto& b : proposals) {
        rects.emplace_back(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    }
    
    // NMS（非极大值抑制）
    std::vector<int> indices;
    cv::dnn::NMSBoxes(rects, confidences, 0.5f, 0.4f, indices);

    std::vector<BBox> final_boxes;
    final_boxes.reserve(indices.size());
    for (int idx : indices) {
        final_boxes.push_back(proposals[idx]);
    }
    return final_boxes;
}

// ----- 核心推理接口，包含耗时诊断（仅输出一次）-----
std::vector<BBox> YOLOInfer::Infer(const cv::Mat& image) {
    if (!initialized_) {
        std::cerr << "Infer called but object is not initialized" << std::endl;
        return {};
    }

    // 静态变量：只在第一次调用时初始化，之后累计耗时
    static int call_count = 0;
    static std::chrono::microseconds pre_sum{0}, infer_sum{0}, post_sum{0};
    static constexpr int kDiagWarmup = 20;   // 预热次数
    static constexpr int kDiagRuns   = 100;  // 统计次数

    auto t0 = std::chrono::high_resolution_clock::now();
    Preprocess(image);
    auto t1 = std::chrono::high_resolution_clock::now();
    DoInference();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto boxes = Postprocess();
    auto t3 = std::chrono::high_resolution_clock::now();

    ++call_count;
    // 跳过前 kDiagWarmup 次预热，之后累计 kDiagRuns 次统计
    if (call_count > kDiagWarmup && call_count <= kDiagWarmup + kDiagRuns) {
        pre_sum   += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        infer_sum += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        post_sum  += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
    }
    // 统计满 kDiagRuns 次后，输出一次平均耗时（微秒）
    if (call_count == kDiagWarmup + kDiagRuns) {
        std::cout << "[Profile] Preprocess avg: " << pre_sum.count() / kDiagRuns
                  << " us, Inference avg: " << infer_sum.count() / kDiagRuns
                  << " us, Postprocess avg: " << post_sum.count() / kDiagRuns
                  << " us" << std::endl;
    }

    return boxes;
}
