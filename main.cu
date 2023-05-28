#include "cuda_fp16.h"
#include "cudaUtil.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cuda_runtime_api.h>

#define MOD 0xFFFFFFFF00000001
#define ROOT 17492915097719143606

template<typename Scalar>
_CUDA_GENERAL_CALL_ Scalar nextPow2(Scalar x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/**
* 大数乘法
*/
_CUDA_GENERAL_CALL_ uint64_t modularMultiplication(uint64_t a, uint64_t b)
{

#ifdef __CUDA_ARCH__  // CUDA设备端实现
    uint64_t resultLow, resultHigh;
    asm volatile("mul.lo.u64 %0, %1, %2;\n" : "=l"(resultLow) : "l"(a), "l"(b));
    asm volatile("mul.hi.u64 %0, %1, %2;\n" : "=l"(resultHigh) : "l"(a), "l"(b));

    uint64_t quotient = (resultHigh / MOD) << 64;
    uint64_t remainder = resultHigh - quotient * MOD;

    quotient += resultLow / MOD;
    remainder = (remainder << 64) + (resultLow - quotient * MOD);

    uint64_t temp = (remainder << 64) + resultLow;
    if (temp >= MOD)
    {
        temp -= MOD;
    }
    temp += quotient * MOD;
    if (temp >= MOD)
    {
        temp -= MOD;
    }

    return temp;
#else  // 主机端实现
    uint64_t result = 0;
    while (a > 0)
    {
        if (a & 1)
        {
            result = (result + b) % MOD;
        }
        a >>= 1;
        b = (b << 1) % MOD;
    }
    return result;
#endif // __CUDA_ARCH__
}

//inline __device__ uint64_t modularMultiplication(uint64_t a, uint64_t b)
//{
//    uint64_t result = 0;
//    a %= MOD; b %= MOD;
//
//    while (b)
//    {
//        if (b & 1) result = (result + a) % MOD;
//
//        a = (a << 1) % MOD;
//        b >>= 1;
//    }
//
//    return result;
//}

/**
* 快速幂
*/
inline _CUDA_GENERAL_CALL_ uint64_t modularExponentiation(uint64_t base, uint64_t exponent)
{
    uint64_t result = 1;
    base %= MOD;

    while (exponent > 0)
    {
        if (exponent & 1) result = modularMultiplication(result, base);

        base = modularMultiplication(base, base);
        exponent >>= 1;
    }

    return result;
}

__global__ void nttKernel(const uint64_t& N, const uint64_t& L, const uint64_t& k, uint64_t* dev_data)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {
        uint64_t m = (1ULL) << k;
        uint64_t m2 = m >> 1;
        uint64_t omega_m = modularExponentiation(ROOT, N / m);

        uint64_t j = idx % m;
        uint64_t i = idx - j;

        uint64_t omega = 1;

        for (uint64_t n = 0; n < m2; n++)
        {
            uint64_t u = dev_data[i + n];
            uint64_t v = modularMultiplication(omega, dev_data[i + n + m2]);

            dev_data[i + n] = (u + v) % MOD;
            dev_data[i + n + m2] = (u - v + MOD) % MOD;

            omega = modularMultiplication(omega, omega_m);
        }
    }
}

namespace {
    // Estimate best block and grid size using CUDA Occupancy Calculator
    int blockSize;   // The launch configurator returned block size 
    int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    int gridSize;    // The actual grid size needed, based on input size 
}

void launchNTT(uint64_t* data, const uint64_t& N)
{
    uint64_t paddedN = nextPow2(N);
    uint64_t L = log2(paddedN);

    uint64_t* dev_data;
    CUDA_CHECK(cudaMalloc((void**)&dev_data, paddedN * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(dev_data, data, N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dev_data + N, 0, (paddedN - N) * sizeof(uint64_t))); // Padding with zeros

    for (uint64_t k = 1; k <= L; k++)
    {
        uint64_t m = (1ULL) << k;
        uint64_t threadsPerBlock = 256;
        uint64_t blocksPerGrid = (paddedN + threadsPerBlock - 1) / threadsPerBlock;

        getOccupancyMaxPotentialBlockSize(paddedN, minGridSize, blockSize, gridSize, nttKernel, 0, 0);
        nttKernel << <gridSize, blockSize >> > (paddedN, L, k, dev_data);
        getLastCudaError("Kernel 'nttKernel' launch failed!\n");
        
        cudaDeviceSynchronize();
    }

    CUDA_CHECK(cudaMemcpy(data, dev_data, paddedN * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_data));
}

void inverseNtt(uint64_t* data, const uint64_t& N)
{
    uint64_t paddedN = nextPow2(N);
    uint64_t L = log2(paddedN);

    uint64_t* dev_data;
    CUDA_CHECK(cudaMalloc((void**)&dev_data, paddedN * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(dev_data, data, paddedN * sizeof(uint64_t), cudaMemcpyHostToDevice));

    for (uint64_t k = L; k >= 1; k--)
    {
        uint64_t m = (1ULL) << k;
        
        getOccupancyMaxPotentialBlockSize(paddedN, minGridSize, blockSize, gridSize, nttKernel, 0, 0);
        nttKernel << <gridSize, blockSize >> > (paddedN, L, k, dev_data);
        getLastCudaError("Kernel 'nttKernel' launch failed!\n");
        
        cudaDeviceSynchronize();
    }

    uint64_t inversedN = modularExponentiation(paddedN, MOD - 2); // Compute modular inverse of paddedN
    uint64_t inversedNMod = modularExponentiation(inversedN, N);

    for (uint64_t i = 0; i < paddedN; i++)
    {
        data[i] = modularMultiplication(data[i], inversedNMod);
    }

    CUDA_CHECK(cudaMemcpy(data, dev_data, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_data));
}

void polynomialMultiply(const uint64_t* coeffA, uint64_t degreeA, const uint64_t* coeffB, uint64_t degreeB, uint64_t* result)
{
    uint64_t degreeLimit = degreeA + degreeB - 1;

    uint64_t paddedDegreeLimit = nextPow2(degreeLimit);
    uint64_t paddedSize = paddedDegreeLimit + 1;

    uint64_t* tempA = new uint64_t[paddedSize];
    uint64_t* tempB = new uint64_t[paddedSize];

    std::fill(tempA, tempA + paddedSize, 0);
    std::fill(tempB, tempB + paddedSize, 0);

    std::copy(coeffA, coeffA + degreeA + 1, tempA);
    std::copy(coeffB, coeffB + degreeB + 1, tempB);

    launchNTT(tempA, paddedSize);
    launchNTT(tempB, paddedSize);

    for (uint64_t i = 0; i < paddedSize; i++)
    {
        tempA[i] = modularMultiplication(tempA[i], tempB[i]);
    }

    inverseNtt(tempA, paddedSize);

    std::copy(tempA, tempA + degreeLimit + 1, result);

    delete[] tempA;
    delete[] tempB;
}

int main(int argc, char** argv)
{
    uint64_t degreeA = 2;
    uint64_t degreeB = 3;

    uint64_t coeffA[] = { 1, 2, 1 }; // A(x) = x^2 + 2x + 1
    uint64_t coeffB[] = { 3, 4, 5, 6 }; // B(x) = 6x^3 + 5x^2 + 4x + 3

    uint64_t degreeLimit = degreeA + degreeB - 1;
    uint64_t* result = new uint64_t[degreeLimit + 1];

    polynomialMultiply(coeffA, degreeA, coeffB, degreeB, result);

    std::cout << "Result of polynomial multiplication:" << std::endl;
    for (uint64_t i = 0; i <= degreeLimit; i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    delete[] result;

    return 0;
}