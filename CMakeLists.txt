# ============================================================
# Copyright (c) 2026 FutureDriver
# SPDX-License-Identifier: MIT
#
# 文件：CMakeLists.txt
# 功能：项目构建配置，编译推理引擎、演示程序和性能测试
# 作者：FutureDriver
# 日期：2026-05-16
# ============================================================

# ============================================================
#  0. 基础设置
# ============================================================
cmake_minimum_required(VERSION 3.18)

# 说明：项目使用 C++ 和 CUDA，因为预处理包含 .cu 文件和 GPU kernel
project(YOLOTRTDemo LANGUAGES CXX CUDA)

# 强制 C++17：智能指针、string_view、if 初始化等现代特性需要
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ============================================================
#  1. 依赖库配置
# ============================================================

# ----- 1.1 CUDA 工具链 -----
set(CUDA_INCLUDE_DIR /usr/local/cuda-12.2/targets/x86_64-linux/include)
set(CUDA_LIB_DIR     /usr/local/cuda-12.2/targets/x86_64-linux/lib)

# CUDA Runtime（cudaStreamCreate 等）
set(CUDART_LIB ${CUDA_LIB_DIR}/libcudart.so)

# NPP 库（GPU resize 需要用 nppiResize_8u_C3R）
find_library(NPPI_LIB nppig PATHS ${CUDA_LIB_DIR} REQUIRED)
find_library(NPPC_LIB nppc PATHS ${CUDA_LIB_DIR} REQUIRED)

# ----- 1.2 TensorRT 推理引擎 -----
set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIB_DIR     /usr/lib/x86_64-linux-gnu)

find_library(NVINFER_LIB nvinfer
             PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser
             PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
             PATHS ${TENSORRT_LIB_DIR})

# ----- 1.3 OpenCV（图像读写、NMS）-----
find_package(OpenCV REQUIRED)
# 自动设置 OpenCV_INCLUDE_DIRS 和 OpenCV_LIBS

# ============================================================
#  2. 公共编译配置
# ============================================================

# 2.1 所有目标共享的头文件路径
set(COMMON_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include   # 我们自己的 yolo_infer.hpp
)

# 2.2 多数目标需要的公共链接库
set(COMMON_LIBS
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${CUDART_LIB}
    ${OpenCV_LIBS}
    ${NPPI_LIB}
    ${NPPC_LIB}
    pthread
    dl
)

# ============================================================
#  3. 构建目标
# ============================================================

# ----- 3.1 推理核心库 -----
add_library(yolo_infer src/yolo_infer.cpp src/preprocess.cu)
target_include_directories(yolo_infer PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(yolo_infer ${COMMON_LIBS})
# 说明：静态库，被 run_demo 和 benchmark 链接

# ----- 3.2 引擎构建工具 -----
add_executable(build_engine src/build_engine.cpp)
target_include_directories(build_engine PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(build_engine
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${CUDART_LIB}
    ${OpenCV_LIBS}
    pthread
    dl
)
# 说明：不需要 NPP 和 nvinfer_plugin，但为统一保留 plugin

# ----- 3.3 快速演示程序 -----
add_executable(run_demo src/main.cpp)
target_include_directories(run_demo PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(run_demo yolo_infer ${COMMON_LIBS})

# ----- 3.4 性能基准测试 -----
add_executable(benchmark src/benchmark.cpp)
target_include_directories(benchmark PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(benchmark yolo_infer ${COMMON_LIBS})