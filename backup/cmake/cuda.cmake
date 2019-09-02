
find_package(CUDA REQUIRED)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math)    
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lineinfo)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -fPIC)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--default-stream per-thread)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

if(USE_TENSOR_CORE)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_70)
endif()

set(CUDA_LINKER_LIBS "")

if(BUILD_SHARED)
    list(APPEND CUDA_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
    if(USE_CUBLAS)
        list(APPEND CUDA_LINKER_LIBS ${CUDA_CUBLAS_LIBRARIES})
    endif()
else() # BUILD_STATIC
    find_path(CUDA_INCLUDE_DIRS cuda.h PATHS /usr/local/cuda/include
                                             /usr/include)

    if(CUDA_INCLUDE_DIRS)
        include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
        find_library(CUDA_LIBRARY NAMES libcudart_static.a
                                   PATHS ${CUDA_INCLUDE_DIRS}/../lib64/
                                   DOC "library path for cuda.")
        if(CUDA_LIBRARY)
            list(APPEND CUDA_LINKER_LIBS ${CUDA_INCLUDE_DIRS}/../lib64/libcudart_static.a)
        
            if(USE_CUBLAS)
                list(APPEND CUDA_LINKER_LIBS ${CUDA_INCLUDE_DIRS}/../lib64/libcublas_static.a)
                list(APPEND CUDA_LINKER_LIBS ${CUDA_INCLUDE_DIRS}/../lib64/libcublas_device.a)
            endif()
        
        endif()
    endif()
endif()