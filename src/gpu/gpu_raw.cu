#include "include/gpu/gpu_raw.h"

#ifdef USE_GPU

template<typename T>
__global__ void kr_gemm_raw(int m, int n, int k, T* A, int lda, T*B, int ldb, T*C, int ldc, T alpha, T beta){

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(r < m){
        if(c < n){
            T tmp = 0;
            for(int i = 0; i < k; ++i){
                tmp += A[cord(r, i, lda)] * B[cord(i, c, ldb)];
            }

            C[cord(r, c, ldc)] = alpha * tmp + beta * C[cord(r, c, ldc)];
        }
    }
    
}

template<typename T>
void GpuRaw<T>::gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){
    int x_block_size = 32;
    int y_block_size = 16;
    dim3 Dg((m + x_block_size - 1) / x_block_size, (n + y_block_size - 1) / y_block_size);
    dim3 Db(x_block_size, y_block_size);

#ifdef ENABLE_DEBUG
    cudaErrCheck(cudaGetLastError());
#endif

    // warm up
    for(int i = 0; i < Base::_warm_up; ++i){    
        kr_gemm_raw<T><<<Dg, Db, 0, _stream>>>(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
    }

#ifdef ENABLE_DEBUG
    cudaErrCheck(cudaStreamSynchronize(_stream));
    cudaErrCheck(cudaGetLastError());
#endif

    Timer<NV> t(_stream);
    for(int i = 0; i < Base::_iter_num; ++i){
        t.start();
        kr_gemm_raw<T><<<Dg, Db, 0, _stream>>>(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
        t.end();
    }

#ifdef ENABLE_DEBUG
    cudaErrCheck(cudaStreamSynchronize(_stream));
    cudaErrCheck(cudaGetLastError());
#endif

    Base::_elapsed = t.getAverageTimeMs();
    
}

template<typename T>
GpuRaw<T>::GpuRaw(int warm_up, int iter_num)
        :GemmBase<T>(warm_up, iter_num){
    cudaErrCheck(cudaStreamCreate(&_stream));
}

template<typename T>
GpuRaw<T>::~GpuRaw(){
    cudaErrCheck(cudaStreamDestroy(_stream));
}

template<typename T>
void GpuRaw<T>::operator()(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta){

    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_c = nullptr;
    
    int size_a = lda * k;
    int size_b = ldb * n;
    int size_c = ldc * n;
    
    cudaErrCheck(cudaMallocHost((void**)&d_a, size_a * sizeof(T)));
    cudaErrCheck(cudaMallocHost((void**)&d_b, size_b * sizeof(T)));
    cudaErrCheck(cudaMallocHost((void**)&d_c, size_c * sizeof(T)));

    cudaErrCheck(cudaMemcpyAsync(d_a, a, size_a * sizeof(T), cudaMemcpyHostToDevice, _stream));
    cudaErrCheck(cudaMemcpyAsync(d_b, b, size_b * sizeof(T), cudaMemcpyHostToDevice, _stream));
    cudaErrCheck(cudaMemcpyAsync(d_c, c, size_c * sizeof(T), cudaMemcpyHostToDevice, _stream));

    gemm(transA, transB, m, n, k, d_a, lda, d_b, ldb, d_c, ldc, alpha, beta);
    
    cudaErrCheck(cudaMemcpyAsync(c, d_c, size_c * sizeof(T), cudaMemcpyDeviceToHost, _stream));

    cudaErrCheck(cudaStreamSynchronize(_stream));

    cudaErrCheck(cudaFreeHost(d_a));
    cudaErrCheck(cudaFreeHost(d_b));
    cudaErrCheck(cudaFreeHost(d_c));
}

// template instantiation declarations
template class GpuRaw<float>;

// register CPU_RAW to GEMM Repo;
REGISTER_GEMM(GPU_RAW, GpuRaw);

#endif //USE_GPU