# ============================================================
#  CMake 最低版本要求
# ============================================================
cmake_minimum_required(VERSION 3.18)

# ============================================================
#  项目名称 & 使用的编程语言（这里只写 C++）
# ============================================================
project(YOLOTRTDemo LANGUAGES CXX)

# ============================================================
#  强制使用 C++17 标准（现代 C++ 特性需要）
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # 如果编译器不支持 C++17 会直接报错

# ============================================================
#  1. 配置 CUDA 头文件路径
#  TensorRT 会用到 cuda_runtime_api.h，所以必须告诉编译器去哪找
# ============================================================
set(CUDA_INCLUDE_DIR /usr/local/cuda-12.2/targets/x86_64-linux/include)
# 说明：这个路径是我们在容器内用 find 命令找到的真实位置，
# 它包含了所有 CUDA 运行时 API 的头文件。

# ============================================================
#  2. 配置 TensorRT 的路径
#  TensorRT 是 NVIDIA 的推理优化引擎，需要链接它的 C++ 库
# ============================================================
set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)   # 头文件位置
set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu)           # 库文件位置

# 查找具体的 TensorRT 库文件（相当于告诉链接器“这些库必须存在”）
find_library(NVINFER_LIB nvinfer                    # 核心推理库
             PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser          # ONNX 解析器库（读 .onnx 用）
             PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin      # 插件库（FP16等高级特性用）
             PATHS ${TENSORRT_LIB_DIR})

# ============================================================
#  2.5 配置 CUDA Runtime 库（cudart）
#  cudaStreamCreate 等 API 的实现需要 cudart
# ============================================================
set(CUDA_LIB_DIR /usr/local/cuda-12.2/targets/x86_64-linux/lib)
set(CUDART_LIB ${CUDA_LIB_DIR}/libcudart.so)        # 直接使用绝对路径

# ============================================================
#  3. 配置 OpenCV
#  OpenCV 是图像处理库，需要用它在 C++ 里读写图片、做预处理
# ============================================================
find_package(OpenCV REQUIRED)
# 上面的命令会自动找到 OpenCV 的头文件和库，
# 并把路径存到 OpenCV_INCLUDE_DIRS 和 OpenCV_LIBS 变量里

# ============================================================
#  4. 公共包含目录设置（所有目标共享）
# ============================================================
# 把所有可能用到的头文件目录集中到一个变量里，便于维护
set(COMMON_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIR}         # CUDA 头文件
    ${TENSORRT_INCLUDE_DIR}     # TensorRT 头文件
    ${OpenCV_INCLUDE_DIRS}      # OpenCV 头文件
    ${CMAKE_SOURCE_DIR}/include # 本工程的头文件（如 yolo_infer.h）
)

# ============================================================
#  5. 定义通用的链接库列表（给所有目标用）
# ============================================================
set(COMMON_LIBS
    ${NVINFER_LIB}              # 核心推理
    ${NVINFER_PLUGIN_LIB}       # 插件支持
    ${OpenCV_LIBS}              # OpenCV 所有库
    pthread                     # POSIX 线程库
    dl                          # 动态链接库加载
    ${CUDART_LIB}               # CUDA 运行时库（cudaMemcpy 等）（绝对路径，避免搜索问题）
)

# ============================================================
#  6. 构建目标
# ============================================================

# ----- 6.1 推理核心库 yolo_infer -----
add_library(yolo_infer src/yolo_infer.cpp)
target_include_directories(yolo_infer PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(yolo_infer ${COMMON_LIBS})
# 注意：这是一个静态库（默认），会被后续可执行文件链接

# ----- 6.2 引擎构建工具 build_engine -----
add_executable(build_engine src/build_engine.cpp)
target_include_directories(build_engine PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(build_engine
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}     # 构建引擎需要 ONNX 解析器
    ${NVINFER_PLUGIN_LIB}
    ${CUDART_LIB}           # 构建引擎也需要 CUDA 运行时库（因为可能会调用 cudaMalloc 等 API）
    ${OpenCV_LIBS}
    pthread
    dl
)

# ----- 6.3 快速演示程序 run_demo -----
add_executable(run_demo src/main.cpp)
target_include_directories(run_demo PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(run_demo yolo_infer ${COMMON_LIBS})

# ----- 6.4 性能基准测试 benchmark -----
add_executable(benchmark src/benchmark.cpp)
target_include_directories(benchmark PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(benchmark yolo_infer ${COMMON_LIBS})
