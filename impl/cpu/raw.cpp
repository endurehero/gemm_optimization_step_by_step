#include "impl/cpu/gemm_cpu.h"

#ifdef USE_RAW

/**
 * / brief implement gemm without any optimization. 
 */
void gemm_cpu(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta){ 

    for(int col = 0; col < n; ++col){
        for(int row = 0; row < m; ++row){
            float tmp = 0.0;
            for(int p = 0; p < k; ++p){
                tmp += a[cord(row, k, _lda)] * b[cord(k, col, ldb)];
            }
            c[cord(row, col, ldc)] = alpha * tmp + beta * c[cord(row, col, ldc)];
        }
    }
}

#endif