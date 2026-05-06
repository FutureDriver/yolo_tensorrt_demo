#include "yolo_infer.h"
#include <iostream>

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
                  << ") conf=" << b.confidence << " label=" << b.label << std::endl;
    }
    return 0;
}
