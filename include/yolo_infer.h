// ===================================================================
// yolo_infer.h
// 高性能 YOLOv8 TensorRT 推理类 —— 你的面试高光代码
// 核心展示 : RAII 资源管理 | 移动语义 | 异步 CUDA 流 | 智能指针
// ===================================================================
#pragma once

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

// ----- 检测结果结构体 -----
struct BBox {
    float x1, y1, x2, y2;  // 边界框坐标 (左上角 & 右下角)
    float conf;             // 置信度
    int label;              // 类别 ID
};

// ----- 日志记录器（简单终端输出）-----
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 只输出警告及以上级别的信息，减少干扰
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YOLOInfer {
public:
    // ----- 1. 构造函数（加载引擎，分配资源）-----
    explicit YOLOInfer(const std::string& engine_path);

    // ----- 2. 析构函数（自动释放所有 GPU 资源）-----
    ~YOLOInfer();

    // ----- 3. 禁止拷贝，允许移动（面试考点：移动语义）-----
    YOLOInfer(const YOLOInfer&) = delete;
    YOLOInfer& operator=(const YOLOInfer&) = delete;
    YOLOInfer(YOLOInfer&& other) noexcept;
    YOLOInfer& operator=(YOLOInfer&& other) noexcept;

    // ----- 4. 核心推理接口（输入 OpenCV 图片，返回检测框）-----
    std::vector<BBox> Infer(const cv::Mat& image);

    // ----- 5. 性能辅助：获取输入张量尺寸 -----
    int GetInputWidth()  const { return input_w_; }
    int GetInputHeight() const { return input_h_; }

private:
    // ----- 内部流水线函数 -----
    void Preprocess(const cv::Mat& image);        // 图像预处理（CPU->GPU 拷贝）
    void DoInference();                           // 执行推理
    std::vector<BBox> Postprocess();              // 解析输出 + NMS

    // ----- 资源管理（全部用智能指针 RAII 管理）-----
    // TensorRT 对象的专用删除器（面试考点：自定义 deleter）
    struct TrtDeleter {
        template <typename T>
        void operator()(T* p) const {
            if (p) p->destroy();
        }
    };

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;         // 运行时
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;       // 引擎
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_; // 执行上下文

    // CUDA 流（异步操作）
    cudaStream_t stream_ = nullptr;

    // 输入输出缓冲区 (GPU 显存) —— 用 unique_ptr + cudaFree 作为 deleter
    struct CudaDeleter {
        void operator()(void* p) const {
            if (p) cudaFree(p);
        }
    };
    std::unique_ptr<void, CudaDeleter> input_gpu_;
    std::unique_ptr<void, CudaDeleter> output_gpu_;

    // 输入输出在 CPU 端的拷贝
    std::vector<float> input_cpu_;   // 预处理后的数据（归一化 float）
    std::vector<float> output_cpu_;  // 推理输出（从 GPU 拷贝回来）

    size_t input_size_{0};   // 输入缓冲区字节数
    size_t output_size_{0};  // 输出缓冲区字节数
    int input_w_{640}, input_h_{640}; // 模型输入尺寸

    // 绑定缓冲区指针（TensorRT API 要求）
    std::vector<void*> buffers_;
    bool initialized_ = false; // 初始化成功标志

    // 预处理常量
    const float kNormScale = 1.0f / 255.0f;
};
