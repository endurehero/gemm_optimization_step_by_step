
#include "impl/gpu/gemm_gpu.h"

#ifdef USE_CUBLAS
#include"cublas_v2.h"

#define CUBLAS_CHECK(condition) \
    do{\
        cublasStatus_t status = condition;\
        if(condition != CUBLAS_STATUS_SUCCESS){\
            cout << "line " << __LINE__ << ", cublas error!" << endl;\
            exit(1);\
        }\
    } while(0)

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
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    t.start();
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k, &alpha, a, ldb, b, lda, &beta, c, ldc));
    t.end();
    cout << "gpu elapsed time : " << t.elapsed() << " ms,  GFLOPS: " << gflops(2 * m * n * k, t.elapsed()) << endl;
    
    cudaMemcpy(*_C_Dev_Host, d_c, size_c * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Gpu processed!" << std::endl;


    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

#endif