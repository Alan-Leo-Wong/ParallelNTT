#pragma once

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
#define _CUDA_GENERAL_CALL_ __host__ __device__
#else
#define _CUDA_GENERAL_CALL_
#endif // __CUDA_ARCH__

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
inline void __getLastCudaError(const char* errorMessage, const char* file, const int line)
{
    const cudaError_t error_code = cudaGetLastError();

    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "%s(%d) : getLastCudaError() CUDA Error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(error_code), cudaGetErrorString(error_code));
        exit(EXIT_FAILURE);
    }
}

template<class T>
static inline __host__ void getOccupancyMaxPotentialBlockSize(const size_t& dataSize,
    int& minGridSize,
    int& blockSize,
    int& gridSize,
    T      func,
    size_t dynamicSMemSize = 0,
    int    blockSizeLimit = 0)
{
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, blockSizeLimit);
    gridSize = (dataSize + blockSize - 1) / blockSize;
}