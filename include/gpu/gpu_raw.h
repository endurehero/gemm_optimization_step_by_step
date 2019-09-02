#ifndef INCLUDE_GPU_GPU_RAW_H
#define INCLUDE_GPU_GPU_RAW_H
#include "include/GemmFactory.hpp"

template<typename T>
class GpuRaw : public GemmBase<T>{
public:
    typedef GemmBase<T> Base;


    GpuRaw(int warm_up = 0, int iter_num = 1);
    
    virtual ~GpuRaw() override;

    virtual void operator()(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta) override;

private:
    virtual void gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta) override;


    cudaStream_t _stream;
};

#endif //INCLUDE_GPU_GPU_RAW_H