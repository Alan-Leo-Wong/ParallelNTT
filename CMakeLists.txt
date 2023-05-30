cmake_minimum_required(VERSION 3.22)
project(GPU_NTT CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(GPU_NTT main.cu)

set_target_properties(GPU_NTT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_compile_options(--device-int128)
