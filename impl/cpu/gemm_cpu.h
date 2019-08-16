#ifndef IMPL_CPU_GEMM_CPU_H
#define IMPL_CPU_GEMM_CPU_H

#include "common.h"

void gemm_cpu(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta);

#endif