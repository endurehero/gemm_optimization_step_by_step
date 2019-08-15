#include "impl/cpu/gemm_cpu.h"

#ifdef USE_REDUCE_INDEX_OVERHEAD

/**
 * / brief implement gemm with reducing index overhead.
 */
void gemm_cpu(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta){ 
    
    float tmp_c = 0.0;
    for(int col = 0; col < n; ++col){
        for(int row = 0; row < m; ++row){
            float* a_head = &(a[cord(row, 0, lda)]);
            float* b_head = &(b[cord(0, col, ldb)]);
            tmp_c = 0.0;
            
            for(int p = 0; p < k; ++p){
                tmp_c += a_head[p * lda] * b_head[p];
            }
            
            c[cord(row, cols, ldc)] = alpha * tmp_c + beta * c[cord(row, cols, ldc)];
        }
    }
}

#endif