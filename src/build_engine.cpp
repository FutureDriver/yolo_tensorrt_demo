// ============================================================
// Copyright (c) 2026 FutureDriver
// SPDX-License-Identifier: MIT
//
// 文件：build_engine.cpp
// 功能：从 ONNX 构建 FP16 / INT8 TensorRT 引擎
// 作者：FutureDriver
// 日期：2026-05-06
// 修改：2026-05-24 添加 INT8 量化支持
// ============================================================

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

// ----- TensorRT 日志记录器 -----
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};

// ----- INT8 校准器 -----
// 实现 IInt8EntropyCalibrator2 接口，负责向 TensorRT 提供校准数据
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
            std::cerr << "Failed to load calibration image: " << image_paths_[current_index_] << std::endl;
            return false;
        }

        // 预处理：BGR→RGB，resize 到 640x640，归一化，HWC→CHW（必须与推理时完全一致）
        cv::Mat rgb, resized;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(input_w_, input_h_));
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        std::vector<float> cpu_data(input_w_ * input_h_ * 3);
        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);
        int plane_size = input_w_ * input_h_;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(cpu_data.data() + c * plane_size, channels[c].data, plane_size * sizeof(float));
        }

        // 拷贝到 GPU（bindings[0] 指向设备内存）
        cudaMemcpy(bindings[0], cpu_data.data(), cpu_data.size() * sizeof(float), cudaMemcpyHostToDevice);

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

// ============================================================
//  主函数：构建引擎
//  用法：./build_engine [--int8]
//        无参数 → 构建 FP16 引擎
//        --int8  → 构建 INT8 引擎（需要 data/calibration/ 下的校准图片）
// ============================================================
int main(int argc, char** argv) {
    Logger logger;

    // 解析命令行参数，决定构建模式
    bool use_int8 = false;
    if (argc >= 2 && std::string(argv[1]) == "--int8") {
        use_int8 = true;
    }

    // 1. 创建 builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) { std::cerr << "Failed to create builder" << std::endl; return -1; }

    // 2. 创建网络（显式 batch 模式）
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) return -1;

    // 3. 创建 ONNX 解析器
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) return -1;
    // 使用挂载到容器内的绝对路径，指向 models 目录下的 ONNX
    if (!parser->parseFromFile("/workspace/demo/models/yolov8n.onnx",
                               static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        return -1;
    }

    // 4. 构建配置（根据模式设置量化标志）
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GB

    if (use_int8) {
        // INT8 量化模式：同时开启 FP16 可让部分层保留 FP16 精度，提升性能
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Building INT8 engine with calibration..." << std::endl;
    } else {
        // 默认 FP16 模式
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Building FP16 engine..." << std::endl;
    }

    // 添加优化配置文件（即使静态尺寸也必须，否则 TensorRT 会禁用部分优化）
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("images", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 640, 640});
    profile->setDimensions("images", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 640, 640});
    profile->setDimensions("images", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 640, 640});
    config->addOptimizationProfile(profile);

    // 5. INT8 校准（仅在 INT8 模式下）
    std::unique_ptr<Int8Calibrator> calibrator;
    if (use_int8) {
        // 收集校准图片路径
        std::vector<std::string> calib_images;
        cv::glob("/workspace/demo/data/calibration/*.jpg", calib_images, false);
        if (calib_images.empty()) {
            std::cerr << "No calibration images found in data/calibration/" << std::endl;
            return -1;
        }
        calibrator = std::make_unique<Int8Calibrator>(calib_images, 640, 640);
        config->setInt8Calibrator(calibrator.get());
    }

    // 6. 构建序列化引擎
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config)
    );
    if (!plan) { std::cerr << "Engine build failed" << std::endl; return -1; }

    // 7. 写入文件（输出到 models 目录，文件名根据模式变化）
    std::string engine_path = use_int8 ? "/workspace/demo/models/yolov8n_int8.engine"
                                       : "/workspace/demo/models/yolov8n_fp16.engine";
    std::ofstream out(engine_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    std::cout << "Engine built successfully, size: " << plan->size() << " bytes" << std::endl;

    return 0;
}