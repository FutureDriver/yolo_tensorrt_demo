// ============================================================
// Copyright (c) 2026 FutureDriver
// SPDX-License-Identifier: MIT
//
// 文件：preprocess.cu
// 功能：GPU 预处理（NPP resize + CUDA kernel 颜色转换与归一化）
// 作者：FutureDriver
// 日期：2026-05-12
// ============================================================

#include <cuda_runtime_api.h>
#include <nppi_geometry_transforms.h>   // NPP resize 头文件

// ---------- 自定义 kernel：BGR 8-bit → RGB float NCHW ----------
__global__ void bgr8_to_rgb_float_nchw_kernel(
    const uchar3* src,       // 输入 BGR 图像（interleaved）
    float* dst,              // 输出 NCHW float 缓冲区
    int width,               // 图像宽度
    int height               // 图像高度
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_idx = y * width + x;
    uchar3 pixel = src[src_idx];
    int plane_size = width * height;

    dst[0 * plane_size + y * width + x] = pixel.z / 255.0f;  // R
    dst[1 * plane_size + y * width + x] = pixel.y / 255.0f;  // G
    dst[2 * plane_size + y * width + x] = pixel.x / 255.0f;  // B
}

// ---------- 外部调用接口 ----------
void launch_preprocess(
    const unsigned char* img_data,   // BGR 图像数据指针
    int src_width,                   // 原始宽度
    int src_height,                  // 原始高度
    float* gpu_dst,                  // 输出 NCHW 缓冲区 (GPU, 已经 640x640)
    cudaStream_t stream              // CUDA 流
) {
    constexpr int kDstWidth  = 640;
    constexpr int kDstHeight = 640;

    // 1. 异步上传原始图像到 GPU
    size_t src_size = src_width * src_height * sizeof(uchar3);
    uchar3* gpu_src = nullptr;
    cudaMallocAsync(&gpu_src, src_size, stream);
    cudaMemcpyAsync(gpu_src, img_data, src_size, cudaMemcpyHostToDevice, stream);

    // 2. 分配 resize 目标缓冲区
    size_t dst_size = kDstWidth * kDstHeight * sizeof(uchar3);
    uchar3* gpu_resized = nullptr;
    cudaMallocAsync(&gpu_resized, dst_size, stream);

    // 3. NPP resize （此版本不支持 CUDA 流，完成后需同步一次
    NppiSize src_roi = { .width = src_width, .height = src_height };
    NppiSize dst_roi = { .width = kDstWidth, .height = kDstHeight };
    NppStatus ret = nppiResize_8u_C3R(
        reinterpret_cast<const Npp8u*>(gpu_src),
        src_width * 3,
        src_roi,
        NppiRect{ .x = 0, .y = 0, .width = src_width, .height = src_height },
        reinterpret_cast<Npp8u*>(gpu_resized),
        kDstWidth * 3,
        dst_roi,
        NppiRect{ .x = 0, .y = 0, .width = kDstWidth, .height = kDstHeight },
        NPPI_INTER_LINEAR
    );
    if (ret != NPP_SUCCESS) {
        cudaFreeAsync(gpu_src, stream);
        cudaFreeAsync(gpu_resized, stream);
        return;
    }
    cudaStreamSynchronize(stream);   // 保证 NPP 完成后再启动自定义 kernel

    // 4. 自定义 kernel：BGR→RGB + 归一化 + HWC→CHW
    dim3 block(32, 8);
    dim3 grid((kDstWidth + block.x - 1) / block.x, (kDstHeight + block.y - 1) / block.y);
    bgr8_to_rgb_float_nchw_kernel<<<grid, block, 0, stream>>>(
        gpu_resized, gpu_dst, kDstWidth, kDstHeight
    );

    // 5. 释放临时缓冲区
    cudaFreeAsync(gpu_src, stream);
    cudaFreeAsync(gpu_resized, stream);
}