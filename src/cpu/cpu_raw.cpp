#include "include/cpu/cpu_raw.h"


template<typename T>
void CpuRaw<T>::gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){
    for(int col = 0; col < n; ++col){
        for(int row = 0; row < m; ++row){
            T tmp = 0.0;
            for(int p = 0; p < k; ++p){
                tmp += a[cord(row, p, lda)] * b[cord(p, col, ldb)];
            }
            c[cord(row, col, ldc)] = alpha * tmp + beta * c[cord(row, col, ldc)];
        }
    }
}

template<typename T>
void CpuRaw<T>::operator()(bool transA, bool transB, int m, int n, int k, \
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
template class CpuRaw<float>;

// register CPU_RAW to GEMM Repo;
REGISTER_GEMM(CPU_RAW, CpuRaw);