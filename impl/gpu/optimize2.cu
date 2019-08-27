#include "impl/gpu/gemm_gpu.h"
#include<cuda_runtime.h>
#include "stdio.h"

/**
 *   brief: add workload per thread.
 *        if one thread only compute one element in c matrix, it will take 3 instrinctions as following:
 *            #1  load one element in A from shared mem to register file.
 *            #2  load one element in B from shared mem to register file.
 *            #3  FMA.
 *        try to add workload for one thread to increase the ratio of compute / memory load.   
 */

#ifdef USE_GPU_OPT2

#define MACRO_SIZE 64
#define BLOCK_SIZE 16
#define MICRO_SIZE 4

__global__ void kr_gemm(int m, int n, int k, float* a, int lda, float*b, int ldb, float*c, int ldc, float alpha, float beta){

    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    int row_in_block = threadIdx.x;
    int col_in_block = threadIdx.y;
    
    float c_ans[MICRO_SIZE * MICRO_SIZE];
#pragma unroll
    for(int i = 0; i < MICRO_SIZE * MICRO_SIZE; ++i){
        c_ans[i] = 0.0;
    }
    __shared__ float As[MACRO_SIZE][MACRO_SIZE];
    __shared__ float Bs[MACRO_SIZE][MACRO_SIZE];

    float* sub_c = &c[cord(block_row * MACRO_SIZE, block_col * MACRO_SIZE, ldc)];
    float* sub_a = &a[cord(block_row * MACRO_SIZE, 0, lda)];
    float* sub_b = &b[cord(0, block_col * MACRO_SIZE, ldb)];

    for(int p = 0; p < k / MACRO_SIZE; ++p){
        float* sub_sub_a = &sub_a[cord(0, p * MACRO_SIZE, lda)];
        float* sub_sub_b = &sub_b[cord(p * MACRO_SIZE, 0, ldb)];
        
        
        for(int t = 0; t < MICRO_SIZE; ++t){
            // A first row
            As[col_in_block * MICRO_SIZE + t][row_in_block * MICRO_SIZE] =     sub_sub_a[cord(row_in_block * MICRO_SIZE, col_in_block * MICRO_SIZE + t, lda)];
            // A second row
            As[col_in_block * MICRO_SIZE + t][row_in_block * MICRO_SIZE + 1] = sub_sub_a[cord(row_in_block * MICRO_SIZE + 1, col_in_block * MICRO_SIZE + t, lda)];
            // A third row
            As[col_in_block * MICRO_SIZE + t][row_in_block * MICRO_SIZE + 2] = sub_sub_a[cord(row_in_block * MICRO_SIZE + 2, col_in_block * MICRO_SIZE + t, lda)];
            // A fouth row
            As[col_in_block * MICRO_SIZE + t][row_in_block * MICRO_SIZE + 3] = sub_sub_a[cord(row_in_block * MICRO_SIZE + 3, col_in_block * MICRO_SIZE + t, lda)];
            

            // B first col
            Bs[col_in_block * MICRO_SIZE][row_in_block * MICRO_SIZE + t] =     sub_sub_b[cord(row_in_block * MICRO_SIZE + t, col_in_block * MICRO_SIZE, ldb)];
            // B second col
            Bs[col_in_block * MICRO_SIZE + 1][row_in_block * MICRO_SIZE + t] = sub_sub_b[cord(row_in_block * MICRO_SIZE + t, col_in_block * MICRO_SIZE + 1, ldb)];
            // B third col
            Bs[col_in_block * MICRO_SIZE + 2][row_in_block * MICRO_SIZE + t] = sub_sub_b[cord(row_in_block * MICRO_SIZE + t, col_in_block * MICRO_SIZE + 2, ldb)];
            // B fouth col
            Bs[col_in_block * MICRO_SIZE + 3][row_in_block * MICRO_SIZE + t] = sub_sub_b[cord(row_in_block * MICRO_SIZE + t, col_in_block * MICRO_SIZE + 3, ldb)];

        }

        __syncthreads();
        
        
        int start_row = row_in_block * MICRO_SIZE;
        int start_col = col_in_block * MICRO_SIZE;
        
        for(int i = 0; i < MACRO_SIZE; ++i){
            for(int t = 0; t < MICRO_SIZE; ++t){
                //first row
                c_ans[cord(0, t, MICRO_SIZE)] += As[i][start_row] * Bs[start_col + t][i];
                //second row
                c_ans[cord(1, t, MICRO_SIZE)] += As[i][start_row + 1] * Bs[start_col + t][i];
                //third row
                c_ans[cord(2, t, MICRO_SIZE)] += As[i][start_row + 2] * Bs[start_col + t][i];
                //fouth row
                c_ans[cord(3, t, MICRO_SIZE)] += As[i][start_row + 3] * Bs[start_col + t][i];
                
            }
        }

        __syncthreads();
        
    }

    // store c ans from register file to global mem
    float* sub_sub_c = &sub_c[cord(row_in_block * MICRO_SIZE, col_in_block * MICRO_SIZE, ldc)];
    for(int i = 0; i < MICRO_SIZE; ++i){
        sub_sub_c[cord(0, i, ldc)] = alpha * c_ans[cord(0, i, MICRO_SIZE)] + beta * sub_sub_c[cord(0, i, ldc)];
        sub_sub_c[cord(1, i, ldc)] = alpha * c_ans[cord(1, i, MICRO_SIZE)] + beta * sub_sub_c[cord(1, i, ldc)];
        sub_sub_c[cord(2, i, ldc)] = alpha * c_ans[cord(2, i, MICRO_SIZE)] + beta * sub_sub_c[cord(2, i, ldc)];
        sub_sub_c[cord(3, i, ldc)] = alpha * c_ans[cord(3, i, MICRO_SIZE)] + beta * sub_sub_c[cord(3, i, ldc)];
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

    dim3 Dg((m + MACRO_SIZE - 1) / MACRO_SIZE, (n + MACRO_SIZE - 1) / MACRO_SIZE);
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