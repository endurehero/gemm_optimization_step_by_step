#if 0
#include "impl/gpu/gemm_gpu.h"

#ifdef USE_CUBLAS

void gemm_gpu(int m, int n, int k, float*a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta, float** _C_Dev_Host){

    Timer<NV> t;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    
    int size_a = m * k;
    int size_b = k * n;
    int size_c = m * n;
    
    cudaMalloc((void**)&d_a, size_a * sizeof(float));
    cudaMalloc((void**)&d_b, size_b * sizeof(float));
    cudaMalloc((void**)&d_c, size_c * sizeof(float));
    *_C_Dev_Host = (float*)(malloc(size_c * sizeof(float)));
    memset(*_C_Dev_Host, 0, size_c * sizeof(float));
    
    cudaMemcpy(d_a, a, size_a * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size_c * sizeof(float), cudaMemcpyHostToDevice);
    
    

    cublasHandle_t cublasHandle;
    cublasErrCheck(cublasCreate(&cublasHandle));
    t.start();
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                m, n, k, &alpha, 
                                d_a, CUDA_R_32F, lda,
                                d_b, CUDA_R_32F, ldb, &beta,
                                d_c, CUDA_R_32F, ldc,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    t.end();


    std::cout << "gpu elapsed time : " << t.elapsed() << " ms,  GFLOPS: " << gflops(2 * m * n * k, t.elapsed()) << std::endl;
    
    cudaMemcpy(*_C_Dev_Host, d_c, size_c * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Gpu processed!" << std::endl;


    cublasErrCheck(cublasDestroy(cublasHandle));
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

#endif
#endif