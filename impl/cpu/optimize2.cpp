#include "impl/cpu/gemm_cpu.h"

#ifdef USE_COL_UNROLL

/**
 * / brief implement gemm with reducing index overhead and unroll the col loop.
 */

void addDot4(int k, float* a, int lda, float*b, int ldb, float* c, int ldc, float alpha, float beta){
    
    float c_0 = 0.0, c_1 = 0.0, c_2 = 0.0, c_3 =0.0;
    
    for(int p = 0;  p < k; ++p){
        float a_p = a[p * lda];
        c_0 += a_p * b[p];
        c_1 += a_p * b[ldb + p];
        c_2 += a_p * b[2 * ldb + p];
        c_3 += a_p * b[3 * ldb + p];
    }

    *c = alpha * c_0 + beta * *c; c += ldc;
    *c = alpha * c_1 + beta * *c; c += ldc;
    *c = alpha * c_2 + beta * *c; c += ldc;
    *c = alpha * c_3 + beta * *c;
}

void gemm_cpu(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta){ 
    
    for(int col = 0; col < n; col += 4){
        for(int row = 0; row < m; ++row){
            
            float* a_head = &(a[cord(row, 0, lda)]);
            float* b_head = &(b[cord(0, col, ldb)]);
            float* c_head = &(c[cord(row, col, ldc)]);
            addDot4(k, a_head, lda, b_head, ldb, c_head, ldc, alpha, beta);
        }
    }
    
}


#endif