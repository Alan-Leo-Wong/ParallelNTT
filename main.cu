#include "NTT.hpp"
#include "config.hpp"
#include <string>
#include <iostream>
#include <algorithm>

int main(int argc, char **argv) {
    int numIters = 5;
    int n = 5, m = 5;
    TEST_TYPE test_type = TEST_TYPE::CUDA;
    if (argc > 1) {
        std::string arg1 = argv[1];
        std::transform(arg1.begin(), arg1.end(), arg1.begin(), ::tolower);  // 将参数转换为小写字母
        if (arg1 == "cpu") test_type = TEST_TYPE::CPU;
        else if(arg1 == "simd") test_type = TEST_TYPE::SIMD;
        if (argc > 2) n = std::stoi(argv[2]);
        if (argc > 3) m = std::stoi(argv[3]);
        if (argc > 4) numIters = std::stoi(argv[4]);
    }
    std::cout << "-- \033[0m\033[1;36m[INFO]\033[0m Using " << testTypeToString(test_type) << std::endl;
    std::cout << "-- \033[0m\033[1;36m[INFO]\033[0m Iterations = " << numIters << std::endl;
    std::cout << "-- \033[0m\033[1;36m[INFO]\033[0m n(多项式 A 次数的log) = " << n << std::endl;
    std::cout << "-- \033[0m\033[1;36m[INFO]\033[0m m(多项式 B 次数的log) = " << m << std::endl;

    NTT ntt(n, m);
    ntt.run(test_type, numIters);

    return 0;
}