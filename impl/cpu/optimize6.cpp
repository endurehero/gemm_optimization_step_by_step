#include "impl/cpu/gemm_cpu.h"

#ifdef USE_BLOCK4X4_VREG

/**
 * / brief implement gemm with Block4X4 and unroll the col loop and more 128bits register.
 */

#include <mmintrin.h>
#include <xmmintrin.h> // SSE
#include <pmmintrin.h> // SSE2
#include <emmintrin.h> // SSE3
typedef union{
    __m128 v;
    float d[4];
}v4f_t;

void addDot4X4VReg(int k, float* a, int lda, float*b, int ldb, float* c, int ldc, float alpha, float beta){
    v4f_t
        c_col0_vreg, c_col1_vreg, c_col2_vreg, c_col3_vreg,
        a_p_vreg, b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

    float
        *b_p0_ptr = b,
        *b_p1_ptr = b + ldb,
        *b_p2_ptr = b + 2 * ldb,
        *b_p3_ptr = b + 3 * ldb;
    
    c_col0_vreg.v = _mm_setzero_ps();
    c_col1_vreg.v = _mm_setzero_ps();
    c_col2_vreg.v = _mm_setzero_ps();
    c_col3_vreg.v = _mm_setzero_ps();

    for(int p = 0; p < k; ++p){
        a_p_vreg.v = _mm_load_ps(static_cast<float*>(a)); a += lda;
        b_p0_vreg.v = _mm_load1_ps(static_cast<float*>(b_p0_ptr++));
        b_p1_vreg.v = _mm_load1_ps(static_cast<float*>(b_p1_ptr++));
        b_p2_vreg.v = _mm_load1_ps(static_cast<float*>(b_p2_ptr++));
        b_p3_vreg.v = _mm_load1_ps(static_cast<float*>(b_p3_ptr++));

        c_col0_vreg.v += a_p_vreg.v * b_p0_vreg.v;
        c_col1_vreg.v += a_p_vreg.v * b_p1_vreg.v;
        c_col2_vreg.v += a_p_vreg.v * b_p2_vreg.v;
        c_col3_vreg.v += a_p_vreg.v * b_p3_vreg.v;
    }

    // first col
    c[cord(0, 0, ldc)] = alpha * c_col0_vreg.d[0] + beta * c[cord(0, 0, ldc)];
    c[cord(1, 0, ldc)] = alpha * c_col0_vreg.d[1] + beta * c[cord(1, 0, ldc)];
    c[cord(2, 0, ldc)] = alpha * c_col0_vreg.d[2] + beta * c[cord(2, 0, ldc)];
    c[cord(3, 0, ldc)] = alpha * c_col0_vreg.d[3] + beta * c[cord(3, 0, ldc)];
    // second col
    c[cord(0, 1, ldc)] = alpha * c_col1_vreg.d[0] + beta * c[cord(0, 1, ldc)];
    c[cord(1, 1, ldc)] = alpha * c_col1_vreg.d[1] + beta * c[cord(1, 1, ldc)];
    c[cord(2, 1, ldc)] = alpha * c_col1_vreg.d[2] + beta * c[cord(2, 1, ldc)];
    c[cord(3, 1, ldc)] = alpha * c_col1_vreg.d[3] + beta * c[cord(3, 1, ldc)];
    // third col
    c[cord(0, 2, ldc)] = alpha * c_col2_vreg.d[0] + beta * c[cord(0, 2, ldc)];
    c[cord(1, 2, ldc)] = alpha * c_col2_vreg.d[1] + beta * c[cord(1, 2, ldc)];
    c[cord(2, 2, ldc)] = alpha * c_col2_vreg.d[2] + beta * c[cord(2, 2, ldc)];
    c[cord(3, 2, ldc)] = alpha * c_col2_vreg.d[3] + beta * c[cord(3, 2, ldc)];
    // fouth
    c[cord(0, 3, ldc)] = alpha * c_col3_vreg.d[0] + beta * c[cord(0, 3, ldc)];
    c[cord(1, 3, ldc)] = alpha * c_col3_vreg.d[1] + beta * c[cord(1, 3, ldc)];
    c[cord(2, 3, ldc)] = alpha * c_col3_vreg.d[2] + beta * c[cord(2, 3, ldc)];
    c[cord(3, 3, ldc)] = alpha * c_col3_vreg.d[3] + beta * c[cord(3, 3, ldc)];
}


void gemm_cpu(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta){ 
    
    for(int col = 0; col < n; col += 4){
        for(int row = 0; row < m; row += 4){
            
            float* a_head = &(a[cord(row, 0, lda)]);
            float* b_head = &(b[cord(0, col, ldb)]);
            float* c_head = &(c[cord(row, col, ldc)]);
            addDot4X4VReg(k, a_head, lda, b_head, ldb, c_head, ldc, alpha, beta);
        }
    }
    
}


#endif