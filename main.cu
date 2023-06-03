#include "NTT.hpp"
#include "config.hpp"
#include <string>
#include <iostream>
#include <algorithm>

//double
//polynomialMultiply(const bool &useCUDA,
//                   const _uint128_t *coeffA, const _uint128_t &degreeA,
//                   const _uint128_t *coeffB, const _uint128_t &degreeB,
//                   _uint128_t *result) {
//    _uint128_t degreeLimit = degreeA + degreeB;
//    _uint128_t paddedDegreeSize = 1;
//    while (paddedDegreeSize <= degreeLimit) paddedDegreeSize <<= 1, ++L;
//
//    _uint128_t *tempA = new _uint128_t[paddedDegreeSize];
//    _uint128_t *tempB = new _uint128_t[paddedDegreeSize];
//    rev = new _uint128_t[paddedDegreeSize];
//
//    std::fill(tempA, tempA + paddedDegreeSize, 0);
//    std::fill(tempB, tempB + paddedDegreeSize, 0);
//    std::copy(coeffA, coeffA + degreeA + 1, tempA);
//    std::copy(coeffB, coeffB + degreeB + 1, tempB);
//
//    std::fill(rev, rev + paddedDegreeSize, 0);
//    for (int i = 0; i < paddedDegreeSize; i++) {
//        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1));
//    }
//
//    double diff;
//    if (useCUDA) {
//        static double t_start = get_seconds();
//        cu_launchNTT(false, paddedDegreeSize, tempA);
//        cu_launchNTT(false, paddedDegreeSize, tempB);
//        for (int i = 0; i < paddedDegreeSize; ++i) {
//            tempA[i] = ((_uint128_t) tempA[i] * (_uint128_t) tempB[i]) % MOD;
//        }
//        cu_launchNTT(true, paddedDegreeSize, tempA);
//        diff = get_seconds() - t_start;
//        t_start += diff;
//    } else {
//        static double t_start = get_seconds();
//        launchNTT(false, paddedDegreeSize, tempA);
//        launchNTT(false, paddedDegreeSize, tempB);
//        for (int i = 0; i < paddedDegreeSize; ++i) {
//            tempA[i] = ((_uint128_t) tempA[i] * (_uint128_t) tempB[i]) % MOD;
//        }
//        launchNTT(true, paddedDegreeSize, tempA);
//        diff = get_seconds() - t_start;
//        t_start += diff;
//    }
//
//    std::copy(tempA, tempA + degreeLimit + 1, result);
//
//    inv = modularExponentiation(paddedDegreeSize, MOD - 2);
//
//    delete[] tempA;
//    delete[] tempB;
//    delete[] rev;
//
//    return diff;
//}

//void generateInputData(const int &n, const int &m,
//                       const ull &degreeA, const ull &degreeB,
//                       _uint128_t *coeffA, _uint128_t *coeffB) {
//    int coMin = 0, coMax = 9;
//    // 从低到高的系数
//    for (ull i = 0; i <= degreeA; ++i) {
//        int x = coMin + rand() % (coMax - coMin + 1);
//        coeffA[i] = x;
//    }
//    for (ull i = 0; i <= degreeB; ++i) {
//        int x = coMin + rand() % (coMax - coMin + 1);
//        coeffB[i] = x;
//    }
//
//    std::ofstream out("input.txt");
//    if (!out) return;
//    out << n << " " << m << std::endl;
//    for (ull i = 0; i <= degreeA; ++i) {
//        out << (int) (coeffA[i]) << " ";
//    }
//    out << std::endl;
//    for (ull i = 0; i <= degreeB; ++i) {
//        out << (int) (coeffB[i]) << " ";
//    }
//    out.close();
//}

int main(int argc, char **argv) {
    int numIters = 5;
    int n = 0, m = 1;
    TEST_TYPE test_type = CUDA;
    if (argc > 1) {
        std::string arg1 = argv[1];
        std::transform(arg1.begin(), arg1.end(), arg1.begin(), ::tolower);  // 将参数转换为小写字母
        if (argv[1] == "cpu") test_type = CPU;
        else if(argv[1] == "simd") test_type = SIMD;
        if (argc > 2) sscanf(argv[2], "%d", &numIters);
        if (argc > 3) sscanf(argv[2], "%d", &n);
        if (argc > 4) sscanf(argv[3], "%d", &m);
    }
    std::cout << "-- Using " << testTypeToString(test_type) << std::endl;
    std::cout << "-- Iterations = " << numIters << std::endl;
    std::cout << "-- n(多项式 A 次数的log) = " << n << std::endl;
    std::cout << "-- m(多项式 B 次数的log) = " << m << std::endl;

    NTT ntt(n, m);
    ntt.run(test_type, numIters);
//    // 最高次数
//    ull degreeA, degreeB;
//    degreeA = 1 << n; degreeB = 1 << m;
//    _uint128_t *coeffA = new _uint128_t[degreeA + 1];
//    _uint128_t *coeffB = new _uint128_t[degreeB + 1];
//
//    generateInputData(n, m, degreeA, degreeB, coeffA, coeffB);
//
//    _uint128_t degreeLimit = degreeA + degreeB; // 卷积后的最高次数
//    _uint128_t *result = new _uint128_t[degreeLimit + 1];
//
////    double time = polynomialMultiply(useCUDA, coeffA, degreeA, coeffB, degreeB, result);
//
//    std::cout << "-- Time(s): " << time << std::endl;
////    std::cout << "--Result of polynomial multiplication:" << std::endl;
////    for (int i = 0; i <= degreeLimit; i++) {
////        std::cout << (unsigned long long) (((_uint128_t) result[i] * (_uint128_t) inv) % MOD);
////        if (i == degreeLimit) std::cout << std::endl;
////        else std::cout << " ";
////    }
//
//    delete[] coeffA;
//    delete[] coeffB;
//    delete[] result;

    return 0;
}