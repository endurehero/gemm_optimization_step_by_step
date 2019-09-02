#include "include/gpu/gpu_opt1.h"

#ifdef USE_GPU

#define BLOCK_SIZE 16

template<typename T>
__global__ void kr_gemm_opt1(int m, int n, int k, T* a, int lda, T*b, int ldb, T*c, int ldc, T alpha, T beta){
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    int row_in_block = threadIdx.x;
    int col_in_block = threadIdx.y;
    
    int row_in_c = block_row * BLOCK_SIZE;
    int col_in_c = block_col * BLOCK_SIZE;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    T* sub_c = &c[cord(row_in_c, col_in_c, ldc)];
    
    T c_ans = 0.0;
    // block in k axis with BLOCK_SIZE
    for(int p = 0; p  < (k / BLOCK_SIZE); ++p){
        T* sub_a = &(a[cord(row_in_c, p * BLOCK_SIZE, lda)]);
        T* sub_b = &(b[cord(p  * BLOCK_SIZE, col_in_c, ldb)]);    
        
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

template<typename T>
void GpuOpt1<T>::gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){
    dim3 Dg((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 Db(BLOCK_SIZE, BLOCK_SIZE);

    // warm up
    for(int i = 0; i < Base::_warm_up; ++i){
        kr_gemm_opt1<T><<<Dg, Db>>>(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
    }

    Timer<NV> t;
    for(int i = 0; i < Base::_iter_num; ++i){
        t.start();
        kr_gemm_opt1<T><<<Dg, Db>>>(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
        t.end();
    }

    Base::_elapsed = t.getAverageTimeMs();
}

template<typename T>
void GpuOpt1<T>::operator()(bool transA, bool transB, int m, int n, int k, \
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
template class GpuOpt1<float>;

// register CPU_RAW to GEMM Repo;
REGISTER_GEMM(GPU_OPT1, GpuOpt1);

#endif //USE_GPU