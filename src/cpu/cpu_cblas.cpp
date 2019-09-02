#include "include/cpu/cpu_cblas.h"

#ifdef USE_CBLAS
#include <cblas.h>
#endif

template<typename T>
void CpuCblas<T>::gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){
#ifdef USE_CBLAS
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
}

template<typename T>
void CpuCblas<T>::operator()(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){

#ifdef USE_CUBLAS
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
#else
    printf("Please switch USE_CBLAS on.!\n");
#endif
}

// template instantiation declarations
template class CpuCblas<float>;

// register CPU_RAW to GEMM Repo;
REGISTER_GEMM(CPU_CBLAS, CpuCblas);