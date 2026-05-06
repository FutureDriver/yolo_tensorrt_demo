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
find_library(NVINFER_LIB nvinfer             # 核心推理库
             PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser   # ONNX 解析器库（读 .onnx 用）
             PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin # 插件库（FP16等高级特性用）
             PATHS ${TENSORRT_LIB_DIR})

# ============================================================
#  3. 配置 OpenCV
#  OpenCV 是图像处理库，需要用它在 C++ 里读写图片、做预处理
# ============================================================
find_package(OpenCV REQUIRED)
# 上面的命令会自动找到 OpenCV 的头文件和库，
# 并把路径存到 OpenCV_INCLUDE_DIRS 和 OpenCV_LIBS 变量里

# ============================================================
#  4. 定义我们要编译的可执行文件
# ============================================================
add_executable(build_engine src/build_engine.cpp)
# 这里只编译一个文件：src/build_engine.cpp，
# 生成的可执行文件名字叫 build_engine

# ============================================================
#  5. 包含头文件目录（让编译器能找到 #include 的文件）
# ============================================================
target_include_directories(build_engine PRIVATE
    ${CUDA_INCLUDE_DIR}         # CUDA 头文件
    ${TENSORRT_INCLUDE_DIR}     # TensorRT 头文件
    ${OpenCV_INCLUDE_DIRS}      # OpenCV 头文件
)
# PRIVATE 表示这些头文件只在这个目标内部使用，不会传染给其他目标

# ============================================================
#  6. 链接库文件（把代码里用到的函数实现“绑”到可执行文件上）
# ============================================================
target_link_libraries(build_engine
    ${NVINFER_LIB}          # 核心推理
    ${NVONNXPARSER_LIB}     # ONNX 解析
    ${NVINFER_PLUGIN_LIB}   # 插件支持
    ${OpenCV_LIBS}          # OpenCV 所有库
    pthread                 # POSIX 线程库（并行构建可能用到）
    dl                      # 动态链接库加载（dlopen 等）
)
# 链接顺序不重要，只要所有需要的库都在就行
