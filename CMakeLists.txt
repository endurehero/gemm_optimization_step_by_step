project(GEMM)

cmake_minimum_required(VERSION 2.8)

include(cmake/utils.cmake)

option(ENABLE_DEBUG "debug mode" NO)
option(BUILD_SHARED "build gemm implement with shared library, otherwise with static library" NO)
option(USE_GPU "use nvidia gpu to accelerate." NO)

if(USE_GPU)
    include(cmake/cuda.cmake)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(UNIX OR APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

if(ENABLE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

option(USE_RAW "use cpu raw" NO)
option(USE_REDUCE_INDEX_OVERHEAD "use cpu optimize1" NO)
option(USE_COL_UNROLL "use cpu optimize2" NO)
option(USE_SCALAR_REGISTER "use cpu optimize3" NO)
option(USE_BLOCK4X4 "use cpu optimize4" NO)
option(USE_BLOCK4X4_REG "use cpu optimize5" NO)
option(USE_BLOCK4X4_VREG "use cpu optimize6" NO)
option(USE_BLOCK4X4_CACHED_VREG "use cpu optimized7" NO)

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