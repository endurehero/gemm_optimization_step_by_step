#include "include/gpu/gpu_cublas.h"

#ifdef USE_GPU
#ifdef USE_CUBLAS

template<typename T>
void GpuCublas<T>::gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){

    cublasHandle_t cublasHandle;
    cublasErrCheck(cublasCreate(&cublasHandle));

    // warm up
    for(int i = 0; i < Base::_warm_up; ++i){
#ifdef USE_TENSOR_CORE
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                m, n, k, &alpha, 
                                a, CUDA_R_32F, lda,
                                b, CUDA_R_32F, ldb, &beta,
                                c, CUDA_R_32F, ldc,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
#else
        cublasErrCheck(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                m, n, k, &alpha,
                                a, lda, b, ldb, &beta, c, ldc));
#endif
    }

    Timer<NV> t;
    for(int i = 0; i < Base::_iter_num; ++i){
        t.start();
#ifdef USE_TENSOR_CORE
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                m, n, k, &alpha, 
                                a, CUDA_R_32F, lda,
                                b, CUDA_R_32F, ldb, &beta,
                                c, CUDA_R_32F, ldc,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
#else
        cublasErrCheck(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                m, n, k, &alpha,
                                a, lda, b, ldb, &beta, c, ldc));
#endif
        t.end();
    }
    
    Base::_elapsed = t.getAverageTimeMs();
}

template<typename T>
void GpuCublas<T>::operator()(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){

    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_c = nullptr;
    
    int size_a = lda * k;
    int size_b = ldb * n;
    int size_c = ldc * n;
    
    cudaErrCheck(cudaMalloc((void**)&d_a, size_a * sizeof(T)));
    cudaErrCheck(cudaMalloc((void**)&d_b, size_b * sizeof(T)));
    cudaErrCheck(cudaMalloc((void**)&d_c, size_c * sizeof(T)));

    cudaErrCheck(cudaMemcpy(d_a, a, size_a * sizeof(T), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_b, b, size_b * sizeof(T), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_c, c, size_c * sizeof(T), cudaMemcpyHostToDevice));
    
    gemm(transA, transB, m, n, k, d_a, lda, d_b, ldb, d_c, ldc, alpha, beta);
    
    cudaErrCheck(cudaMemcpy(c, d_c, size_c * sizeof(T), cudaMemcpyDeviceToHost));

    cudaErrCheck(cudaFree(d_a));
    cudaErrCheck(cudaFree(d_b));
    cudaErrCheck(cudaFree(d_c));
}

// template instantiation declarations
template class GpuCublas<float>;

// register CPU_RAW to GEMM Repo;
REGISTER_GEMM(GPU_CUBLAS, GpuCublas);

#endif //USE_CUBLASE
#endif //USE_GPU