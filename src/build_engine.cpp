// ============================================================
// Copyright (c) 2026 FutureDriver
// SPDX-License-Identifier: MIT
//
// 文件：build_engine.cpp
// 功能：从 ONNX 构建 FP16 / INT8 TensorRT 引擎，支持端到端检测
// 作者：FutureDriver
// 日期：2026-05-24
// ============================================================

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <NvInfer.h>
#include <NvInferPlugin.h>      // 必须！用于注册 EfficientNMS_TRT 等内置插件
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

// ----- TensorRT 日志记录器 -----
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};

// ----- INT8 校准器 -----
// 注意：当前在 WSL2 环境下构建 INT8 引擎会失败，代码已就绪，待原生环境验证
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8Calibrator(const std::vector<std::string>& image_paths,
                   int input_w, int input_h)
        : image_paths_(image_paths), input_w_(input_w), input_h_(input_h),
          batch_size_(1), input_count_(image_paths.size()) {}

    int getBatchSize() const noexcept override { return batch_size_; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (current_index_ >= input_count_) return false;

        cv::Mat img = cv::imread(image_paths_[current_index_]);
        if (img.empty()) {
            std::cerr << "Failed to load calibration image: "
                      << image_paths_[current_index_] << std::endl;
            return false;
        }

        // 预处理：BGR→RGB，resize，归一化，HWC→CHW
        cv::Mat rgb, resized;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(input_w_, input_h_));
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        std::vector<float> cpu_data(input_w_ * input_h_ * 3);
        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);
        int plane_size = input_w_ * input_h_;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(cpu_data.data() + c * plane_size,
                        channels[c].data, plane_size * sizeof(float));
        }

        // 拷贝到 GPU（bindings[0] 指向设备内存）
        cudaMemcpy(bindings[0], cpu_data.data(),
                   cpu_data.size() * sizeof(float), cudaMemcpyHostToDevice);

        current_index_++;
        return true;
    }

    const void* readCalibrationCache(std::size_t& length) noexcept override { return nullptr; }
    void writeCalibrationCache(const void* cache, std::size_t length) noexcept override {}

private:
    const std::vector<std::string> image_paths_;
    int input_w_, input_h_;
    int batch_size_;
    int current_index_ = 0;
    int input_count_;
};

// ----- 主函数 -----
// 用法：./build_engine        → 构建 FP16 引擎（默认）
//       ./build_engine --int8 → 构建 INT8 引擎（需校准图片）
int main(int argc, char** argv) {
    Logger logger;

    // 初始化 TensorRT 内置插件（C++ API 必须显式调用）
    initLibNvInferPlugins(reinterpret_cast<void*>(&logger), "");

    bool use_int8 = false;
    if (argc >= 2 && std::string(argv[1]) == "--int8") {
        use_int8 = true;
    }

    // 1. 创建 builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    if (!builder) { std::cerr << "Failed to create builder" << std::endl; return -1; }

    // 2. 创建网络
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) return -1;

    // 3. 解析 ONNX
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));
    if (!parser) return -1;
    if (!parser->parseFromFile("/workspace/demo/models/yolov8n.onnx",
                               static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        return -1;
    }

    // 4. 构建配置
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    if (use_int8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Building INT8 engine with calibration..." << std::endl;
    } else {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Building FP16 engine..." << std::endl;
    }

    // 优化配置文件（固定 640x640 输入）
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("images", nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4{1, 3, 640, 640});
    profile->setDimensions("images", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4{1, 3, 640, 640});
    profile->setDimensions("images", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4{1, 3, 640, 640});
    config->addOptimizationProfile(profile);

    // 5. INT8 校准（仅在 --int8 模式下）
    std::unique_ptr<Int8Calibrator> calibrator;
    if (use_int8) {
        std::vector<std::string> calib_images;
        cv::glob("/workspace/demo/data/calibration/*.jpg", calib_images, false);
        if (calib_images.empty()) {
            std::cerr << "No calibration images found in data/calibration/" << std::endl;
            return -1;
        }
        calibrator = std::make_unique<Int8Calibrator>(calib_images, 640, 640);
        config->setInt8Calibrator(calibrator.get());
    }

    // 6. 构建引擎
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan) { std::cerr << "Engine build failed" << std::endl; return -1; }

    // 7. 写入文件
    std::string engine_path = use_int8
        ? "/workspace/demo/models/yolov8n_int8.engine"
        : "/workspace/demo/models/yolov8n_fp16.engine";
    std::ofstream out(engine_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    std::cout << "Engine built successfully, size: " << plan->size() << " bytes" << std::endl;

    return 0;
}