#include<cuda_runtime.h>
#include<stdio.h>
#include "common.h"
#include "impl/gpu/gemm_gpu.h"
#include "timer.h"


__global__ void kr_gemm(int m, int n, int k, float* A, int lda, float*B, int ldb, float*C, int ldc, float alpha, float beta){

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int dimx = blockDim.x;
    int dimy = blockDim.y;
    
    for(int r = tidx; r < m; r += dimx){
        for(int c = tidy; c < n; c += dimy){
            float tmp = 0;
            for(int i = 0; i < k; ++i){
                tmp += A[cord(r, i, lda)] * B[cord(i, c, ldb)];
            }

            C[cord(r, c, ldc)] = alpha * tmp + beta * C[cord(r, c, ldc)];
        }
    }
    
}


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
    
    
    dim3 Dg(1, 1);
    dim3 Db(16, 16);
    
    t.start();
    kr_gemm<<<Dg, Db>>>(m, n, k, d_a, lda, d_b, ldb, d_c, ldc, alpha, beta);
    t.end();
    cout << "gpu elapsed time : " << t.elapsed() << " ms,  GFLOPS: " << gflops(2 * m * n * k, t.elapsed()) << endl;
    
    cudaMemcpy(*_C_Dev_Host, d_c, size_c * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Gpu processed!" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}