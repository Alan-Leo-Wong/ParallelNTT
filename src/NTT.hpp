//
// Created by lei on 2023/6/2.
//

#ifndef GPU_NTT_NTT_HPP
#define GPU_NTT_NTT_HPP

#include "config.hpp"
#include "math_util.hpp"
#include "time_util.hpp"
#include "device_launch_parameters.h"
#include <vector>

class NTT {
private:
    int n, m;
    ull degreeA, degreeB;

private:
    int L;
    _uint128_t inv;
    std::vector<_uint128_t> rev;
//    _uint128_t* rev;

public:
    NTT(const int &_n, const int &_m) : n(_n), m(_m), L(0) {degreeA = 1 << n; degreeB = 1 << m;}
    NTT() : NTT(10, 10) {}

//    ~NTT() {delete[] rev; rev= nullptr;}

private:
    void launch_cpuNTT(const _uint128_t& paddedN,
                       _uint128_t * tempA,
                       _uint128_t * tempB,
                       _uint128_t *result);

    void launch_cuNTT(const _uint128_t& paddedN,
                      _uint128_t * tempA,
                      _uint128_t * tempB,
                      _uint128_t *result);

    void polynomialMultiply(const TEST_TYPE& test_type,
                            const _uint128_t *coeffA,
                            const _uint128_t *coeffB,
                            TimerInterface* timer,
                            std::vector<_uint128_t>& result);

    void generateInputData(const std::string& in_filename, _uint128_t* coeffA, _uint128_t* coeffB) const;

public:
    void run(const TEST_TYPE& type, const int& numIters);
};

#endif //GPU_NTT_NTT_HPP
