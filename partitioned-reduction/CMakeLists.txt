cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project( cuda_example LANGUAGES CUDA CXX )

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(${PROJECT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(partitioned_reduction
        main.cpp
        cuda_functions.cu)