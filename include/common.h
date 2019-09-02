#ifndef INCLUDE_COMMON_H
#define INCLUDE_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include "config.h"


#define ErrCheckExt(stat1, stat2)\
    do{\
        ErrCheckExt_((stat1), (stat2), __FILE__, __LINE__);\
    }while(0)

inline void ErrCheckExt_(bool stat1, bool stat2, const char* file, int line){
    if(stat1 != stat2){
        fprintf(stderr, "Err Occured! File: %s, Line: %d\n", file, line);
        exit(1);
    }
}


#define ErrCheck(stat1, stat2)\
    do{\
        ErrCheck_((stat1), (stat2), __FILE__, __LINE__);\
    }while(0)

inline void ErrCheck_(bool stat1, bool stat2, const char* file, int line){
    if(stat1 != stat2){
        fprintf(stderr, "Err Occured! File: %s, Line: %d\n", file, line);
    }
}


#define CLASS_SINGLETON_DECLARE(classname)\
    classname() = delete;\
    ~classname() = delete;\
    classname(const classname&) = delete;\
    classname& operator=(const classname&) = delete


#define cord(r, c, ld) ((ld) * (c) + (r))
    

template<typename DataType>
void fillRandom(DataType* m, int size);
inline float gflops(long long ins_num, float t);
inline float efficiency(float self, float target);

typedef enum{
    CPU_RAW = 0,
    CPU_CBLAS,
    CPU_OPT1,
    CPU_OPT2,
    CPU_OPT3,
    CPU_OPT4,
    CPU_OPT5,
    CPU_OPT6,
    CPU_OPT7,
    CPU_OPT8,
    CPU_OPT9,
    
    GPU_RAW = 100,
    GPU_CUBLAS,
    GPU_CUBLAS_TENSORCORE,
    GPU_OPT1,
    GPU_OPT2,
    GPU_OPT3
    
}GEMM_ALGO_TYPE;


GEMM_ALGO_TYPE algoName2Type(std::string name);
std::string algoType2Name(GEMM_ALGO_TYPE type);



#ifdef USE_GPU
#include <cuda_runtime.h>

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
inline void cudaErrCheck_(cudaError_t stat, const char* file, int line){
    if(cudaSuccess != stat){
        fprintf(stderr, "CUDA Error: %s, File: %s, Line: %d\n", cudaGetErrorString(stat), file, line); exit(1);
    }
}

#ifdef USE_CUBLAS
#include <cublas_v2.h>

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
inline void cublasErrCheck_(cublasStatus_t stat, const char* file, int line){
    
    if(CUBLAS_STATUS_SUCCESS != stat){
        fprintf(stderr, "CUDA Error: %d, File: %s, Line: %d\n", stat, file, line); exit(1);
    }
    
}
#endif // USE_CUBLAS

#endif // USE_GPU

#endif // INCLUDE_COMMON_H