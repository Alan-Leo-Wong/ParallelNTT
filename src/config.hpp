//
// Created by lei on 2023/6/2.
//

#ifndef GPU_NTT_CONFIG_HPP
#define GPU_NTT_CONFIG_HPP

#include <string>

enum TEST_TYPE {
    CUDA,
    CPU,
    NORMAL
};

inline std::string testTypeToString(const TEST_TYPE& type) {
    switch (type) {
        case CUDA:
            return "CUDA";
        case CPU:
            return "CPU";
        case NORMAL:
            return "NORMAL";
        default:
            return "Unknown";
    }
}

#endif //GPU_NTT_CONFIG_HPP
