project(GEMM)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v;-use_fast_math;-lineinfo;-std=c++11)

option(USE_RAW "use cpu raw" NO)
option(USE_REDUCE_INDEX_OVERHEAD "use cpu optimize1" NO)
option(USE_COL_UNROLL "use cpu optimize2" NO)
option(USE_SCALAR_REGISTER "use cpu optimize3" YES)

configure_file{
    ${GEMM_SOURCE_DIR}/cmake/config.h.in
    ${GEMM_BINARY_DIR}/config.h
}
include_directories(${GEMM_BINARY_DIR})

include_directories(${GEMM_SOURCE_DIR}/impl)


cuda_add_executable(
        GEMM
        main.cpp gemm.cpp common.cpp
)

target_link_libraries(GEMM ${impl})