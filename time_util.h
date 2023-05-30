//
// Created by lei on 2023/5/30.
//

#ifndef GPU_NTT_TIME_UTIL_H
#define GPU_NTT_TIME_UTIL_H

#include <chrono>

double get_seconds() {
    return std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
}

#endif //GPU_NTT_TIME_UTIL_H
