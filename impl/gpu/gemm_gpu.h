#ifndef IMPL_GPU_GEMM_GPU_H
#define IMPL_GPU_GEMM_GPU_H
#include "common.h"
#include "timer.h"



extern "C" {
    void gemm_gpu(int m, int n, int k, float*a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta, float** _C_Dev_Host);
}

#endif