// ============================================================
// Copyright (c) 2026 FutureDriver
// SPDX-License-Identifier: MIT
//
// 文件：benchmark.cpp
// 功能：性能基准测试，对 C++ TensorRT 进行 1000 次推理性能基准测试，
//       并与 Python 基线（CSV 格式）自动对比，输出标准化结果
// 作者：FutureDriver
// 日期：2026-05-11
// ============================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "yolo_infer.hpp"

int main() {
    constexpr std::string_view engine_path = "models/yolov8n_fp16.engine";
    constexpr std::string_view image_path  = "data/demo.jpg";

    // 1. 加载图像
    cv::Mat img = cv::imread(image_path.data());
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << '\n';
        return -1;
    }

    // 2. 初始化推理类
    YOLOInfer infer(engine_path.data());

    // 3. 预热（与 Python 基线一致：50 次）
    std::cout << "Warming up (50 iterations)...\n";
    for (int i = 0; i < 50; ++i) {
        infer.Infer(img);
    }

    // 4. 正式测量 1000 次
    constexpr int num_runs = 1000;
    std::vector<double> latencies;
    latencies.reserve(num_runs);
    std::cout << "Benchmarking " << num_runs << " runs...\n";
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
    double p95 = sorted_lat[static_cast<std::size_t>(num_runs * 0.95)];
    double p99 = sorted_lat[static_cast<std::size_t>(num_runs * 0.99)];
    double fps = 1000.0 / avg;

    // 打印结果
    std::cout << "\n========== C++ TensorRT FP16 Benchmark ==========\n";
    std::cout << "Average latency: " << std::fixed << std::setprecision(2) << avg << " ms\n";
    std::cout << "Std deviation:   " << stddev << " ms\n";
    std::cout << "P95 latency:     " << p95 << " ms\n";
    std::cout << "Throughput:      " << fps << " FPS\n";

    // 6. 写 C++ 结果为 CSV（与 Python 基线格式对齐）
    if (std::ofstream out_csv("results/cpp_benchmark.csv"); out_csv) {
        out_csv << "framework,mean_ms,std_ms,min_ms,max_ms,p95_ms,p99_ms,fps\n";
        out_csv << "TensorRT_FP16," << avg << "," << stddev << ","
                << sorted_lat.front() << "," << sorted_lat.back() << ","
                << p95 << "," << p99 << "," << fps << "\n";
        std::cout << "C++ results saved to results/cpp_benchmark.csv\n";
    }

    // 7. 读取 Python 基线 CSV 进行对比
    if (std::ifstream csv_file("results/baseline_benchmark.csv"); csv_file) {
        std::string line;
        std::getline(csv_file, line); // 跳过标题行
        std::cout << "\n--- Comparison with Python baseline ---\n";
        while (std::getline(csv_file, line)) {
            // 用 string_view 避免拷贝
            std::string_view view(line);
            // 跳过空行和 GPU 信息行
            if (view.empty() || view.find("GPU") != std::string_view::npos) continue;
            // 提取 framework（第一列）和 mean_ms（第二列）
            auto pos1 = view.find(',');
            if (pos1 == std::string_view::npos) continue;
            auto framework = view.substr(0, pos1);
            auto rest = view.substr(pos1 + 1);
            auto pos2 = rest.find(',');
            if (pos2 == std::string_view::npos) continue;
            auto mean_str = rest.substr(0, pos2);
            // 字符串转数字（C++17 from_chars 更安全，但为兼容性用 stod）
            double mean_val = std::stod(std::string(mean_str));
            double speedup = mean_val / avg;
            std::cout << "Python " << framework << " avg: " << mean_val
                      << " ms  =>  speedup: " << speedup << "x\n";
        }
    } else {
        std::cout << "Baseline CSV not found, skipping comparison.\n";
    }

    return 0;
}