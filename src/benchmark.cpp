// ============================================================
// Copyright (c) 2026 FutureDriver
// SPDX-License-Identifier: MIT
//
// 文件：benchmark.cpp
// 功能：性能基准测试，对 C++ TensorRT 进行 1000 次推理性能基准测试，
//       并与 Python 基线（CSV格式）自动对比，输出标准化结果
// 作者：FutureDriver
// 日期：2026-05-10
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "yolo_infer.hpp"

int main() {
    const std::string engine_path = "models/yolov8n_fp16.engine";
    const std::string image_path  = "data/demo.jpg";

    // 1. 加载图像
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    // 2. 初始化推理类
    YOLOInfer infer(engine_path);

    // 3. 预热（与 Python 基线一致：50 次）
    std::cout << "Warming up (50 iterations)..." << std::endl;
    for (int i = 0; i < 50; ++i) {
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

    // 排序以计算百分位
    std::vector<double> sorted_lat = latencies;
    std::sort(sorted_lat.begin(), sorted_lat.end());
    double p95 = sorted_lat[static_cast<size_t>(num_runs * 0.95)];
    double p99 = sorted_lat[static_cast<size_t>(num_runs * 0.99)];
    double fps = 1000.0 / avg;

    // 打印结果
    std::cout << "\n========== C++ TensorRT FP16 Benchmark ==========" << std::endl;
    std::cout << "Average latency: " << std::fixed << std::setprecision(2) << avg << " ms" << std::endl;
    std::cout << "Std deviation:   " << stddev << " ms" << std::endl;
    std::cout << "P95 latency:     " << p95 << " ms" << std::endl;
    std::cout << "Throughput:      " << fps << " FPS" << std::endl;

    // 6.  C++ 基准测试结果写入 CSV（与 Python 基线格式对齐）
    std::ofstream out_csv("results/cpp_benchmark.csv");
    out_csv << "framework,mean_ms,std_ms,min_ms,max_ms,p95_ms,p99_ms,fps\n";
    out_csv << "TensorRT_FP16," << avg << "," << stddev << ","
            << sorted_lat.front() << "," << sorted_lat.back() << ","
            << p95 << "," << p99 << "," << fps << "\n";
    out_csv.close();
    std::cout << "C++ results saved to ../results/cpp_benchmark.csv" << std::endl;

    // 7. 读取 Python 基线 CSV 进行对比
    std::ifstream csv_file("results/baseline_benchmark.csv");
    if (csv_file.is_open()) {
        std::string line;
        std::getline(csv_file, line); // 跳过标题行
        std::cout << "\n---------- Comparison with Python baseline ----------" << std::endl;
        while (std::getline(csv_file, line)) {
            if (line.empty() || line.find("GPU") != std::string::npos) continue; // 跳过空白和 GPU 信息行
            std::istringstream ss(line);
            std::string framework, token;
            double mean_val;
            std::getline(ss, framework, ',');
            std::getline(ss, token, ',');   // 读取 mean_ms 列
            mean_val = std::stod(token);
            double speedup = mean_val / avg;
            std::cout << "Python " << framework << " avg: " << mean_val
                      << " ms  =>  speedup: " << speedup << "x" << std::endl;
        }
    } else {
        std::cout << "Baseline CSV not found, skipping comparison." << std::endl;
    }

    return 0;
}