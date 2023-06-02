cmake_minimum_required(VERSION 3.22)
project(GPU_NTT CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(GPU_NTT main.cu NTT.cu)

set_target_properties(GPU_NTT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_compile_options(--device-int128)
