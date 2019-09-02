#include "include/cpu/cpu_opt3.h"

/**
 * / brief implement gemm with reducing index overhead and unroll the col loop and register.
 */
template<typename T>
static void addDot4Reg(int k, T* a, int lda, T*b, int ldb, T* c, int ldc, T alpha, T beta){
    
    register T c_0_reg = 0.0, c_1_reg = 0.0, c_2_reg = 0.0, c_3_reg =0.0;
    register T a_p_reg = 0.0;
    
    T* b_0_ptr = b;
    T* b_1_ptr = b + ldb;
    T* b_2_ptr = b + 2 * ldb;
    T* b_3_ptr = b + 3 * ldb;
    
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

template<typename T>
void CpuOpt3<T>::gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){
    for(int col = 0; col < n; col += 4){
        for(int row = 0; row < m; ++row){
            
            T* a_head = &(a[cord(row, 0, lda)]);
            T* b_head = &(b[cord(0, col, ldb)]);
            T* c_head = &(c[cord(row, col, ldc)]);
            addDot4Reg<T>(k, a_head, lda, b_head, ldb, c_head, ldc, alpha, beta);
        }
    }
}

template<typename T>
void CpuOpt3<T>::operator()(bool transA, bool transB, int m, int n, int k, \
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
template class CpuOpt3<float>;

// register CPU_RAW to GEMM Repo;
REGISTER_GEMM(CPU_OPT3, CpuOpt3);