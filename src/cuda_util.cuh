#pragma once

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
#define _CUDA_GENERAL_CALL_ __host__ __device__
#else
#define _CUDA_GENERAL_CALL_
#endif // __CUDA_ARCH__

#ifndef MIN
#  define MIN(x, y) ((x < y) ? x : y)
#endif // !MIN

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        const cudaError_t error_code = call;                                  \
        if (error_code != cudaSuccess)                                        \
        {                                                                     \
            fprintf(stderr, "CUDA Error:\n");                                          \
            fprintf(stderr, "    --File:       %s\n", __FILE__);                       \
            fprintf(stderr, "    --Line:       %d\n", __LINE__);                       \
            fprintf(stderr, "    --Error code: %d\n", error_code);                     \
            fprintf(stderr, "    --Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                     \
    } while (0);

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
    const cudaError_t error_code = cudaGetLastError();

    if (error_code != cudaSuccess) {
        fprintf(stderr, "%s(%d) : getLastCudaError() CUDA Error :"
                        " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(error_code), cudaGetErrorString(error_code));
        exit(EXIT_FAILURE);
    }
}

inline int getDeviceCount() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
#ifndef NDEBUG
        printf("-- There are no available device(s) that support CUDA\n");
#endif
        exit(EXIT_FAILURE);
    } else {
#ifndef NDEBUG
        printf("-- Detected %d CUDA Capable device(s)\n", deviceCount);
#endif
    }
    return deviceCount;
}

inline int getMaxComputeDevice() {
    int deviceCount = getDeviceCount();
    int maxNumSMs = 0, maxDevice = 0;
    if (deviceCount > 1) {
        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            if (maxNumSMs < prop.multiProcessorCount) {
                maxNumSMs = prop.multiProcessorCount;
                maxDevice = device;
            }
        }
    }
    return maxDevice;
}

template<class T>
static inline __host__ void getOccupancyMaxPotentialBlockSize(const size_t &dataSize,
                                                              int &minGridSize,
                                                              int &blockSize,
                                                              int &gridSize,
                                                              T func,
                                                              size_t dynamicSMemSize = 0,
                                                              int blockSizeLimit = 0) {
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, blockSizeLimit);
    gridSize = (dataSize + blockSize - 1) / blockSize;
}