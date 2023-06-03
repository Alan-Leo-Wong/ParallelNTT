//
// Created by lei on 2023/6/3.
//

#ifndef EVAL_PY_PY_UTIL_HPP
#define EVAL_PY_PY_UTIL_HPP

#include <cstdio>
#include <string>
#include <sstream>
#include <iostream>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#define WIFEXITED(w) (((w) & 0xff) == 0)
#define WEXITSTATUS(w) (((w) >> 8) & 0xff)
#endif

template<typename... Args>
std::string runPythonScriptAndGetBoolValue(const std::string &scriptName, Args... args) {
    std::ostringstream commandStream;
    ((commandStream << " " << args), ...);

    std::string command = "python " + scriptName + commandStream.str();

    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to open pipe for executing Python script.");
    }

    std::ostringstream resultStream;
    char buffer[128];
    std::string result;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        resultStream << buffer;
    }

    int status = pclose(pipe);

    // 解析脚本输出并获取布尔变量值
    if (WIFEXITED(status) && !WEXITSTATUS(status)) {
        return resultStream.str();
    } else {
        throw std::runtime_error("Failed to close pipe.");
    }
}

#endif //EVAL_PY_PY_UTIL_HPP
