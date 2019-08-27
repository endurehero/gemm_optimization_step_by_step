
#include "impl/gpu/gemm_gpu.h"
#include<cuda_runtime.h>

/**
 * /use shared memory
 */
#ifdef USE_GPU_OPT1

#define BLOCK_SIZE 16

__global__ void kr_gemm(int m, int n, int k, float* a, int lda, float*b, int ldb, float*c, int ldc, float alpha, float beta){
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    int row_in_block = threadIdx.x;
    int col_in_block = threadIdx.y;
    
    int row_in_c = block_row * BLOCK_SIZE;
    int col_in_c = block_col * BLOCK_SIZE;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    float* sub_c = &c[cord(row_in_c, col_in_c, ldc)];
    
    float c_ans = 0.0;
    // block in k axis with BLOCK_SIZE
    for(int p = 0; p  < (k / BLOCK_SIZE); ++p){
        float* sub_a = &(a[cord(row_in_c, p * BLOCK_SIZE, lda)]);
        float* sub_b = &(b[cord(p  * BLOCK_SIZE, col_in_c, ldb)]);    
        
        As[col_in_block][row_in_block] = sub_a[cord(row_in_block, col_in_block, lda)];
        Bs[col_in_block][row_in_block] = sub_b[cord(row_in_block, col_in_block, ldb)];
        
        __syncthreads();
        
        for(int i = 0; i < BLOCK_SIZE; ++i){
            c_ans += As[i][row_in_block] * Bs[col_in_block][i];
        }

        __syncthreads();
    }

    sub_c[cord(row_in_block, col_in_block, ldc)] = alpha * c_ans + beta * sub_c[cord(row_in_block, col_in_block, ldc)];

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

    dim3 Dg((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 Db(BLOCK_SIZE, BLOCK_SIZE);
    
    t.start();
    kr_gemm<<<Dg, Db>>>(m, n, k, d_a, lda, d_b, ldb, d_c, ldc, alpha, beta);
    t.end();
    std::cout << "gpu elapsed time : " << t.elapsed() << " ms,  GFLOPS: " << gflops(2 * m * n * k, t.elapsed()) << std::endl;
    
    cudaMemcpy(*_C_Dev_Host, d_c, size_c * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Gpu processed!" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
#endif