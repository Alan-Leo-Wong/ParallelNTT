cmake_minimum_required(VERSION 3.22)
project(CUDA_NTT CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(${PROJECT_NAME} src/main.cu src/NTT.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP)
if (OpenMP_FOUND)
    message(STATUS "OpenMP_CXX_FLAGS = ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -lomp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_compile_options(--device-int128) # 必须放在末尾
