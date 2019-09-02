#include "include/cpu/cpu_opt2.h"

/**
 * / brief implement gemm with reducing index overhead and unroll the col loop.
 */
template<typename T>
static void addDot4(int k, T* a, int lda, T*b, int ldb, T* c, int ldc, T alpha, T beta){
    
    T c_0 = 0.0, c_1 = 0.0, c_2 = 0.0, c_3 =0.0;
    
    for(int p = 0;  p < k; ++p){
        T a_p = a[p * lda];
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

template<typename T>
void CpuOpt2<T>::gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){
    for(int col = 0; col < n; col += 4){
        for(int row = 0; row < m; ++row){
            
            T* a_head = &(a[cord(row, 0, lda)]);
            T* b_head = &(b[cord(0, col, ldb)]);
            T* c_head = &(c[cord(row, col, ldc)]);
            addDot4<T>(k, a_head, lda, b_head, ldb, c_head, ldc, alpha, beta);
        }
    }
}

template<typename T>
void CpuOpt2<T>::operator()(bool transA, bool transB, int m, int n, int k, \
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
template class CpuOpt2<float>;

// register CPU_RAW to GEMM Repo;
REGISTER_GEMM(CPU_OPT2, CpuOpt2);