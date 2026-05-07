#include "yolo_infer.h"
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <opencv2/dnn.hpp>      // NMS

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

// ----- 图像预处理 -----
void YOLOInfer::Preprocess(const cv::Mat& image) {
    cv::Mat rgb, resized;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    // 使用双线性插值，缩放至模型输入尺寸
    cv::resize(rgb, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, CV_32FC3, kNormScale);

    // HWC -> CHW 存入连续内存
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    float* ptr = input_cpu_.data();
    for (int c = 0; c < 3; ++c) {
        std::memcpy(ptr + c * input_w_ * input_h_,
                    channels[c].data,
                    input_w_ * input_h_ * sizeof(float));
    }

    // 异步拷贝到 GPU
    cudaMemcpyAsync(input_gpu_.get(), input_cpu_.data(), input_size_,
                    cudaMemcpyHostToDevice, stream_);
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

// ----- 核心推理接口，输出各阶段耗时 -----
std::vector<BBox> YOLOInfer::Infer(const cv::Mat& image) {
    if (!initialized_) {
        std::cerr << "Infer called but object is not initialized" << std::endl;
        return {};
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    Preprocess(image);
    auto t1 = std::chrono::high_resolution_clock::now();
    DoInference();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto boxes = Postprocess();
    auto t3 = std::chrono::high_resolution_clock::now();

    // 为了避免 bench 时输出干扰，注释掉日志输出
    //std::cout << "[Latency] pre: "
    //          << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
    //          << " us, inference: "
    //        << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
    //        << " us, post: "
    //        << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count()
    //        << " us" << std::endl;
    return boxes;
}
