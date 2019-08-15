#include "gemm.h"
#include "impl/cpu/gemm_cpu.h"

#ifdef USE_GPU
#include "impl/gpu/gemm_gpu.h"
#endif


void Gemm::gpu(){
#ifdef USE_GPU
    gemm_gpu(_m, _n, _k, _A, _lda, _B, _ldb, _C, _ldc, _alpha, _beta, &_C_Dev_Host);
#endif
}

void Gemm::cpu(){
        int c_size = _m * _n;
        _C_Host = static_cast<float*>(malloc(c_size * sizeof(float)));
        
        memcpy(_C_Host, _C, c_size * sizeof(float));
        gemm_cpu(_m, _n, _k, _A, _lda, _B, _ldb, _C_Host, _ldc, _alpha, _beta);
    }