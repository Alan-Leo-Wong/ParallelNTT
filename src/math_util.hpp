//
// Created by lei on 2023/6/2.
//

#ifndef GPU_NTT_MATH_UTIL_HPP
#define GPU_NTT_MATH_UTIL_HPP

#include "cuda_util.cuh"

#define my_swap(x, y) x ^= y, y ^= x, x ^= y

typedef unsigned long long ull;
typedef unsigned __int128 _uint128_t;

constexpr _uint128_t MOD = 0xFFFFFFFF00000001;
constexpr _uint128_t ROOT = 7;
extern __constant__ _uint128_t d_MOD;
extern __constant__ _uint128_t d_ROOT;

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

#endif //GPU_NTT_MATH_UTIL_HPP
