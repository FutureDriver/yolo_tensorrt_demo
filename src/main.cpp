// ============================================================
// Copyright (c) 2026 FutureDriver
// SPDX-License-Identifier: MIT
//
// 文件：main.cpp
// 功能：快速演示：加载图片，推理并打印检测结果
// 作者：FutureDriver
// 日期：2026-05-06
// ============================================================

#include <iostream>
#include "yolo_infer.hpp"


int main() {
    YOLOInfer infer("../models/yolov8n_fp16.engine");
    cv::Mat img = cv::imread("../data/demo.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }
    auto boxes = infer.Infer(img);
    std::cout << "Detected " << boxes.size() << " objects." << std::endl;
    for (auto& b : boxes) {
        std::cout << "Box: (" << b.x1 << "," << b.y1 << ")-(" << b.x2 << "," << b.y2
                  << ") conf=" << b.conf << " label=" << b.label << std::endl;
    }
    return 0;
}
