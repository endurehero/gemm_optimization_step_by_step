cmake_minimum_required(VERSION 2.8)

project(GEMM C CXX)

set(LINKER_LIBS "")
set(EXEC_OUTPUT ${GEMM_SOURCE_DIR}/output)

include(cmake/utils.cmake)

option(ENABLE_DEBUG "debug mode" NO)
option(BUILD_SHARED "build gemm implement with shared library, otherwise with static library" NO)
option(USE_GPU "use nvidia gpu to accelerate." NO)
option(USE_CBLAS "use cblas libs" NO)
option(USE_OPENMP "use openmp libs" NO)
option(USE_THREADS "use C++11 Thread libs" NO)

if(USE_GPU)
    option(USE_CUBLAS "use cublas lib" NO)
    option(USE_TENSOR_CORE "use tensor core" NO)
    include(cmake/cuda.cmake)
endif()

if(USE_CBLAS)
    include(cmake/altas.cmake)
    include_directories(${ATLAS_INCLUDE_DIR})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")
if(UNIX OR APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

if(ENABLE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if(USE_THREADS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(USE_OPENMP)
    FIND_PACKAGE( OpenMP REQUIRED)
    if(OPENMP_FOUND)
        message("OPENMP FOUND")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

configure_file(
    ${GEMM_SOURCE_DIR}/cmake/config.h.in
    ${GEMM_BINARY_DIR}/config.h
)
include_directories(${GEMM_BINARY_DIR})
include_directories(${GEMM_SOURCE_DIR})

fetch_include_recursively(${GEMM_SOURCE_DIR}/include)
add_subdirectory(${GEMM_SOURCE_DIR}/src)
