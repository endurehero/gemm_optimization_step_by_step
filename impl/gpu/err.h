#ifndef IMPL_GPU_ERR_H
#define IMPL_GPU_ERR_H

#include <cuda_runtime.h>
#include <iostream>

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
inline void cudaErrCheck_(cudaError_t stat, const char* file, int line){
    if(cudaSuccess != stat){
        std::cerr << "CUDA Error: " << cudaGetErrorString(stat) << " " << file << " " << line << std::endl;
    }
}

#ifdef USE_CUBLAS
#include <cublas_v2.h>

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
inline void cublasErrCheck_(cublasStatus_t stat, const char* file, int line){
    
    if(CUBLAS_STATUS_SUCCESS != stat){
        std::cout << "cuBlas Error: " << stat << " " << file << " " << line << std::endl;
    }
    
}
#endif // USE_CUBLAS





#endif