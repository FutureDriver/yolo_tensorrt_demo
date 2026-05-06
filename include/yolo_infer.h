// ===================================================================
// yolo_infer.h
// YOLOv8 TensorRT 高性能推理类
// 展示技术点：RAII 资源管理、移动语义、异步 CUDA 流、自定义删除器
// ===================================================================

#pragma once

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ----- 检测结果结构体 -----
struct BBox {
    float x1, y1, x2, y2;  // 边界框坐标（左上角，右下角）
    float conf;             // 置信度
    int label;              // 类别 ID
};

// ----- 日志记录器（输出警告及以上级别信息）-----
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YOLOInfer {
public:
    // ----- 构造 / 析构 / 移动语义 -----
    explicit YOLOInfer(const std::string& engine_path);
    ~YOLOInfer();

    // 禁止拷贝，允许移动（实现移动构造函数和移动赋值运算符）
    YOLOInfer(const YOLOInfer&) = delete;
    YOLOInfer& operator=(const YOLOInfer&) = delete;
    YOLOInfer(YOLOInfer&& other) noexcept;
    YOLOInfer& operator=(YOLOInfer&& other) noexcept;

    // ----- 核心推理接口 -----
    std::vector<BBox> Infer(const cv::Mat& image);

    // ----- 性能辅助 -----
    int GetInputWidth()  const { return input_w_; }
    int GetInputHeight() const { return input_h_; }

private:
    // 内部流水线
    void Preprocess(const cv::Mat& image);        // 预处理 + 异步上传 GPU
    void DoInference();                           // 执行推理
    std::vector<BBox> Postprocess();              // 解析输出 + NMS

    // ----- 资源管理（RAII + 自定义删除器）-----
    struct TrtDeleter {
        template <typename T>
        void operator()(T* p) const {
            if (p) p->destroy();
        }
    };

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;

    cudaStream_t stream_ = nullptr;

    // GPU 显存缓冲区（自定义删除器调用 cudaFree）
    struct CudaDeleter {
        void operator()(void* p) const {
            if (p) cudaFree(p);
        }
    };
    std::unique_ptr<void, CudaDeleter> input_gpu_;
    std::unique_ptr<void, CudaDeleter> output_gpu_;

    // CPU 端数据缓冲
    std::vector<float> input_cpu_;
    std::vector<float> output_cpu_;

    // 绑定到 TensorRT 执行上下文的缓冲指针数组
    std::vector<void*> buffers_;

    size_t input_size_{0};
    size_t output_size_{0};
    int input_w_{640}, input_h_{640};
    int output_num_anchors_{0};   // 检测头锚点数量
    int output_num_classes_{0};   // 类别数（从引擎推导）

    bool initialized_ = false;

    const float kNormScale = 1.0f / 255.0f;
};
