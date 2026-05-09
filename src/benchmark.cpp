// ============================================================
// Copyright (c) 2026 FutureDriver
// SPDX-License-Identifier: MIT
//
// 文件：benchmark.cpp
// 功能：性能基准测试，对 C++ TensorRT 进行 1000 次推理性能基准测试，并与已保存的 Python 基线比较
// 作者：FutureDriver
// 日期：2026-05-06
// ============================================================

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include "yolo_infer.hpp"

int main() {
    const std::string engine_path = "../models/yolov8n_fp16.engine";
    const std::string image_path = "../data/demo.jpg";
    const std::string baseline_path = "../results/baseline_results.txt";

    // 1. 加载图像
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    // 2. 初始化推理类
    YOLOInfer infer(engine_path);

    // 3. 预热（避免冷启动影响）
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 20; ++i) {
        infer.Infer(img);
    }

    // 4. 正式测量 1000 次
    const int num_runs = 1000;
    std::vector<double> latencies;
    latencies.reserve(num_runs);
    std::cout << "Benchmarking " << num_runs << " runs..." << std::endl;
    for (int i = 0; i < num_runs; ++i) {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto boxes = infer.Infer(img);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        latencies.push_back(elapsed_ms);
    }

    // 5. 计算统计值
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / num_runs;
    double sq_sum = std::inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / num_runs - avg * avg);

    std::cout << "\n========== C++ TensorRT FP16 Benchmark ==========" << std::endl;
    std::cout << "Average latency: " << std::fixed << std::setprecision(2) << avg << " ms" << std::endl;
    std::cout << "Std deviation:   " << std::fixed << std::setprecision(2) << stddev << " ms" << std::endl;

    // 6. 读取 Python 基线，进行对比
    std::ifstream baseline_file(baseline_path);
    if (baseline_file.is_open()) {
        std::string line;
        std::cout << "\n---------- Comparison with Python baseline ----------" << std::endl;
        while (std::getline(baseline_file, line)) {
            // 简单解析：例如 "Ultralytics: avg=5.37ms, std=0.55ms"
            if (line.find("Ultralytics: avg=") != std::string::npos) {
                size_t pos = line.find("avg=");
                std::string val = line.substr(pos + 4);
                double py_avg = std::stod(val);
                std::cout << "Python Ultralytics avg: " << py_avg << " ms  =>  speedup: "
                          << (py_avg / avg) << "x" << std::endl;
            } else if (line.find("ONNX Runtime GPU: avg=") != std::string::npos) {
                size_t pos = line.find("avg=");
                std::string val = line.substr(pos + 4);
                double py_avg = std::stod(val);
                std::cout << "Python ONNX RT       avg: " << py_avg << " ms  =>  speedup: "
                          << (py_avg / avg) << "x" << std::endl;
            }
        }
    } else {
        std::cout << "Baseline file not found, skipping comparison." << std::endl;
    }

    return 0;
}
