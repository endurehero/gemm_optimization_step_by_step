cmake_minimum_required(VERSION 2.8)

project(GEMM)

include(cmake/utils.cmake)

option(ENABLE_DEBUG "debug mode" NO)
option(BUILD_SHARED "build gemm implement with shared library, otherwise with static library" NO)
option(USE_GPU "use nvidia gpu to accelerate." NO)
option(USE_CBLAS "use cblas libs" NO)

if(USE_GPU)
    option(USE_CUBLAS "use cublas lib" NO)
    option(USE_GPU_RAW  "use cpu raw" NO)
    option(USE_GPU_OPT1 "use cpu optimize1" NO)
    option(USE_GPU_OPT2 "use cpu optimize2" NO)
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

option(USE_CPU_RAW  "use cpu raw" NO)
option(USE_CPU_OPT1 "use cpu optimize1" NO)
option(USE_CPU_OPT2 "use cpu optimize2" NO)
option(USE_CPU_OPT3 "use cpu optimize3" NO)
option(USE_CPU_OPT4 "use cpu optimize4" NO)
option(USE_CPU_OPT5 "use cpu optimize5" NO)
option(USE_CPU_OPT6 "use cpu optimize6" NO)
option(USE_CPU_OPT7 "use cpu optimize7" NO)
option(USE_CPU_OPT8 "use cpu optimize8" NO)
option(USE_CPU_OPT9 "use cpu optimize9" NO)


if(USE_CPU_OPT8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(USE_CPU_OPT9)
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
include_directories(${GEMM_SOURCE_DIR}/impl)
add_subdirectory(${GEMM_SOURCE_DIR}/impl)


add_executable(
    GEMM
    main.cpp gemm.cpp common.cpp
)

if(BUILD_SHARED)
    target_link_libraries(GEMM impl)
else()
    target_link_libraries(GEMM -Wl,--whole-archive impl_static -Wl,--no-whole-archive)
endif()