#include "impl/gpu/gemm_gpu.h"
#ifdef USE_TENSOR_CORE
#include<string>
static std::string desc = "\
GPU : TENSOR CORE IMPLEMENTATION\n\
    1. Use wmma to implementation a raw version.\n\
    2. Use raw cublas to be groudtruth.\n\
    3. Use tensor core cublas to be groundtruth.\
";

#include<stdio.h>
#include <mma.h>
#include<cuda_runtime.h>

#ifdef USE_CUBLAS
#include<cublas_v2.h>
#endif

using namespace nvcuda;

#define WARP_SIZE 32

#define BLOCK_SIZE_X 128 // Aligned with warp size
#define BLOCK_SIZE_Y 4   // Max thread num in one block

#define WMMA_MACRO_M 16 // One warp(one thread process 4 elements) process 16 elements for C at m axis.
#define WMMA_MACRO_N 16 // One warp(one thread process 4 elements) process 16 elements for C at n axis.
#define WMMA_MACRO_K 16 // One warp(ont thread process 4 elements) process 16 elements for both A and B at k axis.


__global__ void convertFp32ToFp16(half* out, float* in, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        out[tid] = in[tid];
    }
}


__global__ void kr_gemm(int m, int n, int k, half* a, int lda, half*b, int ldb, float*c, int ldc, float alpha, float beta){
    // Tile using a 2D grid
    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_n = blockIdx.y * blockDim.y + threadIdx.y;


    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_MACRO_M, WMMA_MACRO_N, WMMA_MACRO_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_MACRO_M, WMMA_MACRO_N, WMMA_MACRO_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_MACRO_M, WMMA_MACRO_N, WMMA_MACRO_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_MACRO_M, WMMA_MACRO_N, WMMA_MACRO_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Loop over k
    for(int i = 0; i <= k; i += WMMA_MACRO_K){
        int a_row = warp_m * WMMA_MACRO_M;
        int a_col = i;
        
        int b_row = i;
        int b_col = warp_n * WMMA_MACRO_N;
        
        // bounds checking
        if(a_row < m && a_col < k && b_row < k && b_col < n){
            // load the inputs
            wmma::load_matrix_sync(a_frag, &a[cord(a_row, a_col, lda)], lda);
            wmma::load_matrix_sync(b_frag, &b[cord(b_row, b_col, ldb)], ldb);
            
            // perform
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int c_row = warp_m * WMMA_MACRO_M;
    int c_col = warp_n * WMMA_MACRO_N;
    
    if(c_row < m && c_col < n){
        wmma::load_matrix_sync(c_frag, &c[cord(c_row, c_col, ldc)], ldc, wmma::mem_col_major);
        
        for(int i = 0; i < c_frag.num_elements; ++i){
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // store the output
        wmma::store_matrix_sync(&c[cord(c_row, c_col, ldc)], c_frag, ldc, wmma::mem_col_major);
        
    }

}


void gemm_gpu(int m, int n, int k, float*a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta, float** _C_Dev_Host, std::vector<float>& time, int warm_up, int iter_num){

    std::cout << desc << std::endl;
    
    Timer<NV> t_wmma;

    int size_a = lda * k;
    int size_b = ldb * n;
    int size_c = ldc * n;


    *_C_Dev_Host = (float*)(malloc(size_c * sizeof(float)));
    memset(*_C_Dev_Host, 0, size_c * sizeof(float));

    float* a_fp32 = nullptr;
    float* b_fp32 = nullptr;
    half* a_fp16 = nullptr;
    half* b_fp16 = nullptr;

    float* c_wmma = nullptr;

    cudaErrCheck(cudaMalloc((void**)&a_fp32, size_a * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&b_fp32, size_b * sizeof(float)));
    
    cudaErrCheck(cudaMalloc((void**)&a_fp16, size_a * sizeof(half)));
    cudaErrCheck(cudaMalloc((void**)&b_fp16, size_b * sizeof(half)));

    cudaErrCheck(cudaMalloc((void**)&c_wmma, size_c * sizeof(float)));
    
    cudaErrCheck(cudaMemcpy(a_fp32, a, size_a * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(b_fp32, b, size_b * sizeof(float), cudaMemcpyHostToDevice));
    
    
    convertFp32ToFp16<<<(size_a + 255) / 256, 256>>>(a_fp16, a_fp32, size_a);
    convertFp32ToFp16<<<(size_b + 255) / 256, 256>>>(b_fp16, b_fp32, size_b);
    
    float* c_host_wmma = nullptr;
    c_host_wmma = (float*)malloc(sizeof(float) * size_c);
    
    memcpy(c_host_wmma, c, size_c * sizeof(float));
    cudaErrCheck(cudaMemcpy(c_wmma, c_host_wmma, size_c * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 grid_dim;
    dim3 block_dim;
    
    block_dim.x = BLOCK_SIZE_X;
    block_dim.y = BLOCK_SIZE_Y;
    
    grid_dim.x = (m + (WMMA_MACRO_M * block_dim.x / WARP_SIZE - 1)) / (WMMA_MACRO_M * block_dim.x / WARP_SIZE);
    grid_dim.y = (n + WMMA_MACRO_N * block_dim.y - 1) / (WMMA_MACRO_N * block_dim.y);

    // warm up
    for(int i = 0; i < warm_up; ++i){
        kr_gemm<<<grid_dim, block_dim>>>(m, n, k, a_fp16, lda, b_fp16, ldb, c_wmma, ldc, alpha, beta);    
    }
    
    for(int i = 0; i < iter_num; ++i){
        t_wmma.start();
        kr_gemm<<<grid_dim, block_dim>>>(m, n, k, a_fp16, lda, b_fp16, ldb, c_wmma, ldc, alpha, beta);
        t_wmma.end();    
    }
    printf("wmma elapsed time: %f ms \n", t_wmma.getAverageTimeMs());
    time.emplace_back(t_wmma.getAverageTimeMs());


    //back to host
    cudaErrCheck(cudaMemcpy(*_C_Dev_Host, c_wmma, size_c * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef USE_CUBLAS
    Timer<NV> t_cublas;
    float* c_host_cublas = nullptr;
    c_host_cublas = (float*)malloc(sizeof(float) * size_c);
    memcpy(c_host_cublas, c, size_c * sizeof(float));

    float* c_cublas = nullptr;
    cudaErrCheck(cudaMalloc((void**)&c_cublas, size_c * sizeof(float)));
    cudaErrCheck(cudaMemcpy(c_cublas, c_host_cublas, size_c * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublasHandle;
    cublasErrCheck(cublasCreate(&cublasHandle));
    // warm up
    for(int i = 0; i < warm_up; ++i){
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                m, n, k, &alpha, 
                                a_fp32, CUDA_R_32F, lda,
                                b_fp32, CUDA_R_32F, ldb, &beta,
                                c_cublas, CUDA_R_32F, ldc,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT));    
    }

    for(int i = 0; i < iter_num; ++i){
        t_cublas.start();
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                    m, n, k, &alpha, 
                                    a_fp32, CUDA_R_32F, lda,
                                    b_fp32, CUDA_R_32F, ldb, &beta,
                                    c_cublas, CUDA_R_32F, ldc,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
        t_cublas.end();    
    }
    
    printf("cublas elapsed time: %f ms \n", t_cublas.getAverageTimeMs());
    time.emplace_back(t_cublas.getAverageTimeMs());
    cudaErrCheck(cudaFree(c_cublas));
    free(c_host_cublas);


    // Use cublas with tensor core 
    Timer<NV> t_cublas_tensorcore;
    float* c_host_cublas_tensorcore = nullptr;
    c_host_cublas_tensorcore = (float*)(malloc(size_c * sizeof(float)));
    memcpy(c_host_cublas_tensorcore, c, size_c * sizeof(float));

    float* c_cublas_tensorcore = nullptr;
    cudaErrCheck(cudaMalloc((void**)&c_cublas_tensorcore, size_c * sizeof(float)));
    cudaErrCheck(cudaMemcpy(c_cublas_tensorcore, c_host_cublas_tensorcore, size_c * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublasTensorCoreHandler;
    cublasErrCheck(cublasCreate(&cublasTensorCoreHandler));
    cublasErrCheck(cublasSetMathMode(cublasTensorCoreHandler, CUBLAS_TENSOR_OP_MATH));

    // warm up
    for(int i = 0; i < warm_up; ++i){
        cublasErrCheck(cublasGemmEx(cublasTensorCoreHandler, CUBLAS_OP_N, CUBLAS_OP_N, 
                                m, n, k, &alpha, 
                                a_fp16, CUDA_R_16F, lda,
                                b_fp16, CUDA_R_16F, ldb, &beta,
                                c_cublas_tensorcore, CUDA_R_32F, ldc,
                                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    }


    for(int i = 0; i < iter_num; ++i){
        t_cublas_tensorcore.start();
        cublasErrCheck(cublasGemmEx(cublasTensorCoreHandler, CUBLAS_OP_N, CUBLAS_OP_N, 
                                    m, n, k, &alpha, 
                                    a_fp16, CUDA_R_16F, lda,
                                    b_fp16, CUDA_R_16F, ldb, &beta,
                                    c_cublas_tensorcore, CUDA_R_32F, ldc,
                                    CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        t_cublas_tensorcore.end();
    }
    
    printf("cublas with tensor core elapsed time: %f ms \n", t_cublas_tensorcore.getAverageTimeMs());
    time.emplace_back(t_cublas_tensorcore.getAverageTimeMs());
    cudaErrCheck(cudaFree(c_cublas_tensorcore));
    free(c_host_cublas_tensorcore);
    
#endif

    cudaErrCheck(cudaFree(a_fp32));
    cudaErrCheck(cudaFree(b_fp32));
    cudaErrCheck(cudaFree(a_fp16));
    cudaErrCheck(cudaFree(b_fp16));
    cudaErrCheck(cudaFree(c_wmma));
    free(c_host_wmma);
    
}

#endif