#include <fstream>
#include <iostream>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};

int main() {
    Logger logger;
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

    // 4. 构建配置（开启 FP16，设置最大内存池）
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GB

    // 5. 构建序列化引擎
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config)
    );
    if (!plan) { std::cerr << "Engine build failed" << std::endl; return -1; }

    // 6. 写入文件（输出到 models 目录）
    std::ofstream out("/workspace/demo/models/yolov8n_fp16.engine", std::ios::binary);
    out.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    std::cout << "Engine built successfully, size: " << plan->size() << " bytes" << std::endl;

    return 0;
}
