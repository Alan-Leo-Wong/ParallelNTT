#include "time_util.h"
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

#define my_swap(x, y) x ^= y, y ^= x, x ^= y

typedef unsigned __int128 _uint128_t;

constexpr _uint128_t MOD = 0xFFFFFFFF00000001;
constexpr _uint128_t ROOT = 7;
int L;
_uint128_t inv;
_uint128_t *rev;

__constant__ _uint128_t d_MOD = 0xFFFFFFFF00000001;
__constant__ _uint128_t d_ROOT = 7;
//__constant__ _uint128_t d_ROOT = 17492915097719143606;
__device__ _uint128_t d_r, d_mid, d_wn;

/**
* 快速幂
*/
inline _CUDA_GENERAL_CALL_ _uint128_t modularExponentiation(_uint128_t base, _uint128_t exponent) {
#ifdef __CUDA_ARCH__  // CUDA设备端实现
    _uint128_t result = 1;
    while (exponent > 0) {
        if (exponent & 1) result = (result * base) % d_MOD;

        base = (base * base) % d_MOD;
        exponent >>= 1;
    }
    return result % d_MOD;
#else
    _uint128_t result = 1;
    while (exponent > 0) {
        if (exponent & 1) {
            result = (result * base) % MOD;
        }
//        std::cout << "pre base = " << base << std::endl;
        _uint128_t mul = base * base;
//        std::cout << "mul base = " << mul << std::endl;
        base = mul % MOD;
//        std::cout << "after base = " << base << std::endl;
        exponent >>= 1;
    }
    return result % MOD;
#endif
}

__global__ void nttKernel(const _uint128_t numDivGroups, _uint128_t *d_data) {
    unsigned int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y_idx = threadIdx.y + blockIdx.y * blockDim.y;

    if (x_idx < numDivGroups && y_idx < d_mid) {
        const _uint128_t omega = modularExponentiation(d_wn, y_idx);

        _uint128_t u = d_data[x_idx * d_r + y_idx];
        _uint128_t v = d_data[x_idx * d_r + y_idx + d_mid] * omega % d_MOD;

        d_data[x_idx * d_r + y_idx] = (u + v) % d_MOD;
        d_data[x_idx * d_r + y_idx + d_mid] = (u - v + d_MOD) % d_MOD;
    }
}

void launchNTT(const bool &isInverse, const _uint128_t &paddedN, _uint128_t *data) {
    for (int i = 0; i < paddedN; i++)
        if (i < rev[i]) my_swap(data[i], data[rev[i]]);

    for (int k = 1; k <= L; k++) {
        _uint128_t mid = (1ULL) << (k - 1);

        _uint128_t wn = modularExponentiation(ROOT, ((MOD - 1) >> k));
        if (isInverse) wn = modularExponentiation(wn, MOD - 2);

        for (int j = 0; j < paddedN; j += (mid << 1)) {
            _uint128_t w = 1;
            for (int k = 0; k < mid; k++, w = ((_uint128_t) w * (_uint128_t) wn) % MOD) {
                _uint128_t x = data[j + k], y = ((_uint128_t) w * (_uint128_t) data[j + k + mid]) % MOD;
                data[j + k] = (x + y) % MOD;
                data[j + k + mid] = (x - y + MOD) % MOD;
            }
        }
    }
}

void cu_launchNTT(const bool &isInverse, const _uint128_t &paddedN, _uint128_t *data) {
    for (int i = 0; i < paddedN; i++)
        if (i < rev[i]) my_swap(data[i], data[rev[i]]);

    _uint128_t *d_data;
    CUDA_CHECK(cudaMalloc((void **) &d_data, paddedN * sizeof(_uint128_t)));
    CUDA_CHECK(cudaMemcpy(d_data, data, paddedN * sizeof(_uint128_t), cudaMemcpyHostToDevice));

    dim3 blockSize, gridSize;
    blockSize.x = 8, blockSize.y = 128;
    for (int k = 1; k <= L; k++) {
        _uint128_t mid = (1ULL) << (k - 1);

        CUDA_CHECK(cudaMemcpyToSymbol(d_mid, &mid, sizeof(_uint128_t)));
        _uint128_t wn = modularExponentiation(ROOT, ((MOD - 1) >> k));
        if (isInverse) wn = modularExponentiation(wn, MOD - 2);

        CUDA_CHECK(cudaMemcpyToSymbol(d_wn, &wn, sizeof(_uint128_t)));
        _uint128_t r = mid << 1;
        _uint128_t numDivGroups = (paddedN + r - 1) / r;
        CUDA_CHECK(cudaMemcpyToSymbol(d_r, &r, sizeof(_uint128_t)));

        gridSize.x = (numDivGroups + blockSize.x - 1) / blockSize.x;
        gridSize.y = (mid + blockSize.y - 1) / blockSize.y;

        nttKernel <<<gridSize, blockSize >>>(numDivGroups, d_data);
        getLastCudaError("Kernel 'nttKernel' launch failed!\n");

//        cudaDeviceSynchronize();
    }

    CUDA_CHECK(cudaMemcpy(data, d_data, paddedN * sizeof(_uint128_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

double
polynomialMultiply(const bool &useCUDA,
                   const _uint128_t *coeffA, const _uint128_t &degreeA,
                   const _uint128_t *coeffB, const _uint128_t &degreeB,
                   _uint128_t *result) {
    _uint128_t degreeLimit = degreeA + degreeB;
    _uint128_t paddedDegreeSize = 1;
    while (paddedDegreeSize <= degreeLimit) paddedDegreeSize <<= 1, ++L;

    _uint128_t *tempA = new _uint128_t[paddedDegreeSize];
    _uint128_t *tempB = new _uint128_t[paddedDegreeSize];
    rev = new _uint128_t[paddedDegreeSize];

    std::fill(tempA, tempA + paddedDegreeSize, 0);
    std::fill(tempB, tempB + paddedDegreeSize, 0);
    std::copy(coeffA, coeffA + degreeA + 1, tempA);
    std::copy(coeffB, coeffB + degreeB + 1, tempB);

    std::fill(rev, rev + paddedDegreeSize, 0);
    for (int i = 0; i < paddedDegreeSize; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1));
    }

    double diff;
    if (useCUDA) {
        static double t_start = get_seconds();
        cu_launchNTT(false, paddedDegreeSize, tempA);
        cu_launchNTT(false, paddedDegreeSize, tempB);
        for (int i = 0; i < paddedDegreeSize; ++i) {
            tempA[i] = ((_uint128_t) tempA[i] * (_uint128_t) tempB[i]) % MOD;
        }
        cu_launchNTT(true, paddedDegreeSize, tempA);
        diff = get_seconds() - t_start;
        t_start += diff;
    } else {
        static double t_start = get_seconds();
        launchNTT(false, paddedDegreeSize, tempA);
        launchNTT(false, paddedDegreeSize, tempB);
        for (int i = 0; i < paddedDegreeSize; ++i) {
            tempA[i] = ((_uint128_t) tempA[i] * (_uint128_t) tempB[i]) % MOD;
        }
        launchNTT(true, paddedDegreeSize, tempA);
        diff = get_seconds() - t_start;
        t_start += diff;
    }

    std::copy(tempA, tempA + degreeLimit + 1, result);

    inv = modularExponentiation(paddedDegreeSize, MOD - 2);

    delete[] tempA;
    delete[] tempB;
    delete[] rev;

    return diff;
}

int main(int argc, char **argv) {
    int n = 16, m = 16;
    bool useCUDA = true;
    if (argc > 1) {
        if (argv[1] == "cpu") useCUDA = false;
        if (argc > 2) sscanf(argv[2], "%d", &n);
        if (argc > 3) sscanf(argv[3], "%d", &m);
    }

    // 最高次数
    unsigned long long degreeA, degreeB;
//    std::cin >> degreeA >> degreeB;
    degreeA = 1 << n;
    degreeB = 1 << m;

    _uint128_t *coeffA = new _uint128_t[degreeA + 1];
    _uint128_t *coeffB = new _uint128_t[degreeB + 1];

    // 从低到高的系数
//    for (int i = 0; i <= degreeA; ++i) {
//        unsigned long long x;
//        std::cin >> x;
//        coeffA[i] = x;
//    }
//    for (int i = 0; i <= degreeB; ++i) {
//        unsigned long long x;
//        std::cin >> x;
//        coeffB[i] = x;
//    }
    int coMin = 0, coMax = 9;
    for (int i = 0; i <= degreeA; ++i) {
        int x = coMin + rand() % (coMax - coMin + 1);
        coeffA[i] = x;
    }
    for (int i = 0; i <= degreeB; ++i) {
        int x = coMin + rand() % (coMax - coMin + 1);
        coeffB[i] = x;
    }

    _uint128_t degreeLimit = degreeA + degreeB; // 卷积后的最高次数
    _uint128_t *result = new _uint128_t[degreeLimit + 1];

    double time = polynomialMultiply(useCUDA, coeffA, degreeA, coeffB, degreeB, result);

    std::cout << "--Time(s): " << time << std::endl;
//    std::cout << "--Result of polynomial multiplication:" << std::endl;
//    for (int i = 0; i <= degreeLimit; i++) {
//        std::cout << (unsigned long long) (((_uint128_t) result[i] * (_uint128_t) inv) % MOD);
//        if (i == degreeLimit) std::cout << std::endl;
//        else std::cout << " ";
//    }


    delete[] result;

    return 0;
}