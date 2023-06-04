cmake_minimum_required(VERSION 3.22)
project(CUDA_NTT CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(${PROJECT_NAME} src/main.cu src/NTT.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_compile_options(--device-int128) # 必须放在末尾
