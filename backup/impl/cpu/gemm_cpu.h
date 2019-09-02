#ifndef IMPL_CPU_GEMM_CPU_H
#define IMPL_CPU_GEMM_CPU_H

#include "common.h"

void gemm_cpu(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta);
void gemm_cpu(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc, double alpha, double beta);
#endif