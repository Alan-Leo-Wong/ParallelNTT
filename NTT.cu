//
// Created by lei on 2023/6/2.
//
#include "NTT.hpp"
#include "py_util.hpp"
#include "cuda_fp16.h"
#include "cuda_util.cuh"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <random>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>


void NTT::launch_cpuNTT(const _uint128_t &paddedN,
                        _uint128_t *tempA,
                        _uint128_t *tempB,
                        _uint128_t *result) {
    auto cpuNtt = [&](const bool &isInverse,
                      const _uint128_t &paddedN,
                      _uint128_t *data) {
        for (int i = 0; i < paddedN; i++)
            if (i < rev[i]) my_swap(data[i], data[rev[i]]);

        for (int i = 1; i <= L; ++i) {
            _uint128_t mid = (1ULL) << (i - 1);
            _uint128_t wn = modularExponentiation(ROOT, ((MOD - 1) >> i));
            if (isInverse) wn = modularExponentiation(wn, MOD - 2);

            for (_uint128_t j = 0; j < paddedN; j += (mid << 1)) {
                _uint128_t w = 1;
                for (int k = 0; k < mid; ++k, w = (w * wn) % MOD) {
                    _uint128_t x = data[j + k], y = (w * data[j + k + mid]) % MOD;
                    data[j + k] = (x + y) % MOD;
                    data[j + k + mid] = (x - y + MOD) % MOD;
                }
            }
        }
    };

    cpuNtt(false, paddedN, tempA);
    cpuNtt(false, paddedN, tempB);
    for (int i = 0; i < paddedN; ++i) {
        result[i] = (tempA[i] * tempB[i]) % MOD;
    }
    cpuNtt(true, paddedN, result);
}

namespace {
    __device__ _uint128_t d_r, d_mid, d_wn;
}
__constant__ _uint128_t d_MOD = 0xFFFFFFFF00000001;
__constant__ _uint128_t d_ROOT = 7;
//__constant__ _uint128_t d_ROOT = 17492915097719143606;

__global__ void nttKernel(const _uint128_t numDivGroups, _uint128_t *d_data) {
    unsigned int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y_idx = threadIdx.y + blockIdx.y * blockDim.y;

    if (x_idx < d_mid && y_idx < numDivGroups) {
        const _uint128_t omega = modularExponentiation(d_wn, x_idx);

        _uint128_t u = d_data[y_idx * d_r + x_idx];
        _uint128_t v = d_data[y_idx * d_r + x_idx + d_mid] * omega % d_MOD;

        d_data[y_idx * d_r + x_idx] = (u + v) % d_MOD;
        d_data[y_idx * d_r + x_idx + d_mid] = (u - v + d_MOD) % d_MOD;
    }
}

__global__ void mulKernel(const _uint128_t paddedN,
                          const _uint128_t *d_tempA,
                          const _uint128_t *d_tempB,
                          _uint128_t *d_res) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < paddedN) {
        d_res[idx] = (d_tempA[idx] * d_tempB[idx]) % d_MOD;
    }
}

/**
 * Only to warm-up
 */
__global__ void warmUpKernel() {

}

void NTT::launch_cuNTT(const _uint128_t &paddedN,
                       _uint128_t *tempA,
                       _uint128_t *tempB,
                       _uint128_t *result) {
    auto cuNtt = [&](const bool &isInverse,
                     const _uint128_t &paddedN,
                     _uint128_t *data) {
        for (int i = 0; i < paddedN; ++i)
            if (i < rev[i]) my_swap(data[i], data[rev[i]]);

        _uint128_t *d_data;
        CUDA_CHECK(cudaMalloc((void **) &d_data, paddedN * sizeof(_uint128_t)));
        CUDA_CHECK(cudaMemcpy(d_data, data, paddedN * sizeof(_uint128_t), cudaMemcpyHostToDevice));

        dim3 blockSize, gridSize;
        blockSize.x = 128, blockSize.y = 8;
        for (int k = 1; k <= L; ++k) {
            _uint128_t mid = (1ULL) << (k - 1);

            CUDA_CHECK(cudaMemcpyToSymbol(d_mid, &mid, sizeof(_uint128_t)));
            _uint128_t wn = modularExponentiation(ROOT, ((MOD - 1) >> k));
            if (isInverse) wn = modularExponentiation(wn, MOD - 2);

            CUDA_CHECK(cudaMemcpyToSymbol(d_wn, &wn, sizeof(_uint128_t)));
            _uint128_t r = mid << 1;
            _uint128_t numDivGroups = (paddedN + r - 1) / r;
            CUDA_CHECK(cudaMemcpyToSymbol(d_r, &r, sizeof(_uint128_t)));

            gridSize.y = (numDivGroups + blockSize.y - 1) / blockSize.y;
            gridSize.x = (mid + blockSize.x - 1) / blockSize.x;

            nttKernel<<<gridSize, blockSize>>>(numDivGroups, d_data);
            getLastCudaError("Kernel 'nttKernel' launch failed!\n");
        }

        CUDA_CHECK(cudaMemcpy(data, d_data, paddedN * sizeof(_uint128_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_data));
    };

    cuNtt(false, paddedN, tempA);
    cuNtt(false, paddedN, tempB);

    _uint128_t *d_tempA, *d_tempB, *d_res;
    CUDA_CHECK(cudaMalloc((void **) &d_tempA, sizeof(_uint128_t) * paddedN));
    CUDA_CHECK(cudaMemcpy(d_tempA, tempA, sizeof(_uint128_t) * paddedN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **) &d_tempB, sizeof(_uint128_t) * paddedN));
    CUDA_CHECK(cudaMemcpy(d_tempB, tempB, sizeof(_uint128_t) * paddedN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **) &d_res, sizeof(_uint128_t) * paddedN));

    const int gridSize = (paddedN + 1024 - 1) / 1024;
    mulKernel<<<gridSize, 1024>>>(paddedN, d_tempA, d_tempB, d_res);
    CUDA_CHECK(cudaMemcpy(result, d_res, sizeof(_uint128_t) * paddedN, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_tempA));
    CUDA_CHECK(cudaFree(d_tempB));
    CUDA_CHECK(cudaFree(d_res));

    cuNtt(true, paddedN, result);
}

void NTT::polynomialMultiply(const TEST_TYPE &test_type,
                             const _uint128_t *coeffA,
                             const _uint128_t *coeffB,
                             TimerInterface *timer,
                             std::vector<_uint128_t>& result) {
    _uint128_t degreeLimit = degreeA + degreeB;
    _uint128_t paddedN = 1;
    while (paddedN <= degreeLimit) paddedN <<= 1, ++L;

    auto tempA = new _uint128_t[paddedN];
    auto tempB = new _uint128_t[paddedN];
    rev.clear(); rev.resize(paddedN, 0);
    result.resize(paddedN, 0);

    std::fill(tempA, tempA + paddedN, 0);
    std::fill(tempB, tempB + paddedN, 0);
    std::copy(coeffA, coeffA + degreeA + 1, tempA);
    std::copy(coeffB, coeffB + degreeB + 1, tempB);
//    std::fill(rev, rev + paddedN, 0);
    for (int i = 0; i < paddedN; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1));
    }
    inv = modularExponentiation(paddedN, MOD - 2);

    startTimer(&timer);
    switch (test_type) {
        case CPU:
            launch_cpuNTT(paddedN, tempA, tempB, result.data());
            break;
        case SIMD:

            break;
        default:
            fprintf(stderr, "\033[1;31m[Error]\033[0m Unknown type! Will use CUDA.\n");
        case CUDA:
            launch_cuNTT(paddedN, tempA, tempB, result.data());
            break;
    }
    stopTimer(&timer);

    delete[] tempA;
    delete[] tempB;
}

void NTT::generateInputData(const std::string &in_filename,
                            _uint128_t *coeffA,
                            _uint128_t *coeffB) const {
    int coMin = 0, coMax = 9;
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_int_distribution<int> distribution(coMin, coMax);

    // 从低到高的系数
    for (_uint128_t i = 0; i <= degreeA; ++i) {
        int x = distribution(engine);
        coeffA[i] = x;
    }
    for (_uint128_t i = 0; i <= degreeB; ++i) {
        int x = distribution(engine);
        coeffB[i] = x;
    }

    std::ofstream out(in_filename, std::ios::out);
    if (!out) {
        fprintf(stderr, "[I/O] Line: %d Error: file %s can not be opened!\n", __LINE__, in_filename.c_str());
        return;
    }
    out << degreeA << " " << degreeB << std::endl;
    for (ull i = 0; i <= degreeA; ++i) {
        out << (int) (coeffA[i]) << " ";
    }
    out << std::endl;
    for (ull i = 0; i <= degreeB; ++i) {
        out << (int) (coeffB[i]) << " ";
    }
    out.close();
}

void NTT::run(const TEST_TYPE &type, const int &numIters) {
    const std::string in_filename = "input.txt";
    const std::string res_filename = "result_" + testTypeToString(type) + ".txt";

    TimerInterface *timer;
    createTimer(&timer);

    if (type == TEST_TYPE::CUDA) warmUpKernel<<<1, 1>>>();
    int correct = 0;
    for (int iter = 1; iter <= numIters; ++iter) {
        L = 0;

        auto coeffA = new _uint128_t[degreeA + 1];
        auto coeffB = new _uint128_t[degreeB + 1];
//        coeffA[0] = 1, coeffA[1] = 2;
//        coeffB[0] = 1, coeffB[1] = 2, coeffB[2] = 1;
        generateInputData(in_filename, coeffA, coeffB);

        const _uint128_t degreeLimit = degreeA + degreeB;
        std::vector<_uint128_t> result;

        polynomialMultiply(type, coeffA, coeffB, timer, result);
#ifndef NDEBUG
        printf("\033[1;34m[DEBUG]\033[0m Result of Iter #%d:\n", iter);
        for (_uint128_t iter = 0; iter <= degreeLimit; ++iter)
            std::cout << (ull) ((result[iter] * inv) % MOD) << " ";
        printf("\n==========\n");
#endif
        delete[] coeffA;
        delete[] coeffB;

        std::ofstream out(res_filename, std::ios::out);
        if (!out) {
            fprintf(stderr, "[I/O] Line: %d Error: file %s can not be opened!\n", __LINE__, res_filename.c_str());
            continue;
        }
        for (_uint128_t i = 0; i <= degreeLimit; ++i)
            out << (ull) ((result[i] * inv) % MOD) << " ";
        out.close();

        try {
            std::string scriptName = R"(../eval.py)";

            // 调用Python脚本并获取返回值
            std::string py_res = runPythonScriptAndGetBoolValue(scriptName, in_filename, res_filename);

            bool boolValue = (py_res.find("True") != std::string::npos);
            if (boolValue) {
#ifndef NDEBUG
                printf("-- \033[0m\033[1;36m[INFO]\033[0m"
                       " \033[1;32m[%s]\033[0m"
                       " result at #iter %d is"
                       " \033[1;32mTRUE\033[0m\n",
                       testTypeToString(type).c_str(), iter);
#endif
                ++correct;
            } else {
#ifndef NDEBUG
                printf("-- \033[0m\033[1;36m[INFO]\033[0m"
                       " \033[1;32m[%s]\033[0m"
                       " result at #iter %d is"
                       " \033[1;31mFALSE\033[0m\n",
                       testTypeToString(type).c_str(), iter);
#endif
            }
        } catch (const std::exception &e) {
            std::cerr << "-- \033[0m\033[1;31m[Error]\033[0m " << e.what() << std::endl;
        }
    }
    double avg_time = getAverageTimerValue(&timer) * 1e-3;
    printf("-- \033[0m\033[1;36m[INFO]\033[0m"
           " \033[0m\033[1;32m[%s]\033[0m"
           " %d iterations take an average of"
           " \033[1;31m%lf\033[0m"
           " seconds,"
           " correct rate ="
           " \033[1;31m%.2lf%%\033[0m\n",
           testTypeToString(type).c_str(),
           numIters, avg_time, correct * 100.0 / numIters);

    deleteTimer(&timer);
}