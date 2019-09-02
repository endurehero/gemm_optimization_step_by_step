#include "include/cpu/cpu_opt4.h"

/**
 * / brief implement gemm with Block4X4 and unroll the col loop and register.
 */
template<typename T>
static void addDot4X4Reg(int k, T* a, int lda, T*b, int ldb, T* c, int ldc, T alpha, T beta){
    
    register T 
        c_00_reg = 0.0, c_01_reg = 0.0, c_02_reg = 0.0, c_03_reg = 0.0,
        c_10_reg = 0.0, c_11_reg = 0.0, c_12_reg = 0.0, c_13_reg = 0.0,
        c_20_reg = 0.0, c_21_reg = 0.0, c_22_reg = 0.0, c_23_reg = 0.0,
        c_30_reg = 0.0, c_31_reg = 0.0, c_32_reg = 0.0, c_33_reg = 0.0;
        
    register T 
        a_p0_reg = 0.0,
        a_p1_reg = 0.0,
        a_p2_reg = 0.0,
        a_p3_reg = 0.0;

    T
        *a_p0_ptr = a,
        *a_p1_ptr = a + 1,
        *a_p2_ptr = a + 2,
        *a_p3_ptr = a + 3,
        *b_0p_ptr = b,
        *b_1p_ptr = b + ldb,
        *b_2p_ptr = b + 2 * ldb,
        *b_3p_ptr = b + 3 * ldb;
        

    for(int p = 0; p < k; ++p){
        // first row
        a_p0_reg = a_p0_ptr[p * lda];
        c_00_reg += a_p0_reg * b_0p_ptr[p];
        c_01_reg += a_p0_reg * b_1p_ptr[p];
        c_02_reg += a_p0_reg * b_2p_ptr[p];
        c_03_reg += a_p0_reg * b_3p_ptr[p];

        // second row
        a_p1_reg = a_p1_ptr[p * lda];
        c_10_reg += a_p1_reg * b_0p_ptr[p];
        c_11_reg += a_p1_reg * b_1p_ptr[p];
        c_12_reg += a_p1_reg * b_2p_ptr[p];
        c_13_reg += a_p1_reg * b_3p_ptr[p];
        
        // third row
        a_p2_reg = a_p2_ptr[p * lda];
        c_20_reg += a_p2_reg * b_0p_ptr[p];
        c_21_reg += a_p2_reg * b_1p_ptr[p];
        c_22_reg += a_p2_reg * b_2p_ptr[p];
        c_23_reg += a_p2_reg * b_3p_ptr[p];

        // fouth row
        a_p3_reg = a_p3_ptr[p * lda];
        c_30_reg += a_p3_reg * b_0p_ptr[p];
        c_31_reg += a_p3_reg * b_1p_ptr[p];
        c_32_reg += a_p3_reg * b_2p_ptr[p];
        c_33_reg += a_p3_reg * b_3p_ptr[p];
    }

    // first row
    c[cord(0, 0, ldc)] = alpha * c_00_reg + beta * c[cord(0, 0, ldc)];
    c[cord(0, 1, ldc)] = alpha * c_01_reg + beta * c[cord(0, 1, ldc)];
    c[cord(0, 2, ldc)] = alpha * c_02_reg + beta * c[cord(0, 2, ldc)];
    c[cord(0, 3, ldc)] = alpha * c_03_reg + beta * c[cord(0, 3, ldc)];
    // second row
    c[cord(1, 0, ldc)] = alpha * c_10_reg + beta * c[cord(1, 0, ldc)];
    c[cord(1, 1, ldc)] = alpha * c_11_reg + beta * c[cord(1, 1, ldc)];
    c[cord(1, 2, ldc)] = alpha * c_12_reg + beta * c[cord(1, 2, ldc)];
    c[cord(1, 3, ldc)] = alpha * c_13_reg + beta * c[cord(1, 3, ldc)];
    // third row
    c[cord(2, 0, ldc)] = alpha * c_20_reg + beta * c[cord(2, 0, ldc)];
    c[cord(2, 1, ldc)] = alpha * c_21_reg + beta * c[cord(2, 1, ldc)];
    c[cord(2, 2, ldc)] = alpha * c_22_reg + beta * c[cord(2, 2, ldc)];
    c[cord(2, 3, ldc)] = alpha * c_23_reg + beta * c[cord(2, 3, ldc)];
    // fouth row
    c[cord(3, 0, ldc)] = alpha * c_30_reg + beta * c[cord(3, 0, ldc)];
    c[cord(3, 1, ldc)] = alpha * c_31_reg + beta * c[cord(3, 1, ldc)];
    c[cord(3, 2, ldc)] = alpha * c_32_reg + beta * c[cord(3, 2, ldc)];
    c[cord(3, 3, ldc)] = alpha * c_33_reg + beta * c[cord(3, 3, ldc)];
}


template<typename T>
void CpuOpt4<T>::gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){
    for(int col = 0; col < n; col += 4){
        for(int row = 0; row < m; row += 4){
            
            T* a_head = &(a[cord(row, 0, lda)]);
            T* b_head = &(b[cord(0, col, ldb)]);
            T* c_head = &(c[cord(row, col, ldc)]);
            addDot4X4Reg<T>(k, a_head, lda, b_head, ldb, c_head, ldc, alpha, beta);
        }
    }
}

template<typename T>
void CpuOpt4<T>::operator()(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){

    // warm up
    for(int i = 0; i < Base::_warm_up; ++i){
        gemm(transA, transB, m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
    }

    Timer<CPU> t;
    for(int i = 0; i < Base::_iter_num; ++i){
        t.start();
        gemm(transA, transB, m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
        t.end();
    }

    Base::_elapsed = t.getAverageTimeMs();
}

// template instantiation declarations
template class CpuOpt4<float>;

// register CPU_RAW to GEMM Repo;
REGISTER_GEMM(CPU_OPT4, CpuOpt4);