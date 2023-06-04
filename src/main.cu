#include "NTT.hpp"
#include "config.hpp"
#include <string>
#include <iostream>
#include <algorithm>

int numIters = 5;
int n = 5, m = 5;
TEST_TYPE test_type = TEST_TYPE::CUDA;

// Parse the program parameters and set them as global variables
void parseProgramParameters(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);  // 将参数转换为小写字母

        if (arg == "-cpu") {
            test_type = TEST_TYPE::CPU;
        } else if (arg == "-iter") {
            if (i + 1 < argc) {
                numIters = std::stoi(argv[i + 1]);
                ++i;
            }
        } else if (arg == "-n") {
            if (i + 1 < argc) {
                n = std::stoi(argv[i + 1]);
                ++i;
            }
        } else if (arg == "-m") {
            if (i + 1 < argc) {
                m = std::stoi(argv[i + 1]);
                ++i;
            }
        }
    }
}

int main(int argc, char **argv) {
    parseProgramParameters(argc, argv);

    std::cout << "-- \033[0m\033[1;36m[INFO]\033[0m Using " << testTypeToString(test_type) << std::endl;
    std::cout << "-- \033[0m\033[1;36m[INFO]\033[0m Iterations = " << numIters << std::endl;
    std::cout << "-- \033[0m\033[1;36m[INFO]\033[0m n(多项式 A 次数的log) = " << n << std::endl;
    std::cout << "-- \033[0m\033[1;36m[INFO]\033[0m m(多项式 B 次数的log) = " << m << std::endl;

    NTT ntt(n, m);
    ntt.run(test_type, numIters);

    return 0;
}