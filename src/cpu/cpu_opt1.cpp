#include "include/cpu/cpu_opt1.h"


template<typename T>
void CpuOpt1<T>::gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){
    T tmp_c = 0.0;
    for(int col = 0; col < n; ++col){
        for(int row = 0; row < m; ++row){
            float* a_head = &(a[cord(row, 0, lda)]);
            float* b_head = &(b[cord(0, col, ldb)]);
            tmp_c = 0.0;
            
            for(int p = 0; p < k; ++p){
                tmp_c += a_head[p * lda] * b_head[p];
            }
            
            c[cord(row, col, ldc)] = alpha * tmp_c + beta * c[cord(row, col, ldc)];
        }
    }
}

template<typename T>
void CpuOpt1<T>::operator()(bool transA, bool transB, int m, int n, int k, \
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
template class CpuOpt1<float>;

// register CPU_RAW to GEMM Repo;
REGISTER_GEMM(CPU_OPT1, CpuOpt1);