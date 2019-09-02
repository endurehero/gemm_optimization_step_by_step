#include "impl/cpu/gemm_cpu.h"

#ifdef USE_CPU_OPT3

/**
 * / brief implement gemm with reducing index overhead and unroll the col loop and register.
 */

void addDot4Reg(int k, float* a, int lda, float*b, int ldb, float* c, int ldc, float alpha, float beta){
    
    register float c_0_reg = 0.0, c_1_reg = 0.0, c_2_reg = 0.0, c_3_reg =0.0;
    register float a_p_reg = 0.0;
    
    float* b_0_ptr = b;
    float* b_1_ptr = b + ldb;
    float* b_2_ptr = b + 2 * ldb;
    float* b_3_ptr = b + 3 * ldb;
    
    for(int p = 0;  p < k; ++p){
        a_p_reg = a[p * lda];
        
        c_0_reg += a_p_reg * *b_0_ptr++;
        c_1_reg += a_p_reg * *b_1_ptr++;
        c_2_reg += a_p_reg * *b_2_ptr++;
        c_3_reg += a_p_reg * *b_3_ptr++;
    }

    *c = alpha * c_0_reg + beta * *c; c += ldc;
    *c = alpha * c_1_reg + beta * *c; c += ldc;
    *c = alpha * c_2_reg + beta * *c; c += ldc;
    *c = alpha * c_3_reg + beta * *c;
}

void gemm_cpu(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta){ 
    
    for(int col = 0; col < n; col += 4){
        for(int row = 0; row < m; ++row){
            
            float* a_head = &(a[cord(row, 0, lda)]);
            float* b_head = &(b[cord(0, col, ldb)]);
            float* c_head = &(c[cord(row, col, ldc)]);
            addDot4Reg(k, a_head, lda, b_head, ldb, c_head, ldc, alpha, beta);
        }
    }
    
}


#endif