#include "gemm.h"
#include "timer.h"
#include "impl/cpu/gemm_cpu.h"

#ifdef USE_GPU
#include "impl/gpu/gemm_gpu.h"
#endif

template<typename DataType>
void Gemm<DataType>::gpu(){
#ifdef USE_GPU
    gemm_gpu(_m, _n, _k, _A, _lda, _B, _ldb, _C, _ldc, _alpha, _beta, &_C_Dev_Host);
#endif
}

template<typename DataType>
void Gemm<DataType>::cpu(){
        int c_size = _m * _n;
        _C_Host = static_cast<DataType*>(malloc(c_size * sizeof(DataType)));
        
        memcpy(_C_Host, _C, c_size * sizeof(float));

        Timer<CPU> t_h;
        t_h.start();
        gemm_cpu(_m, _n, _k, _A, _lda, _B, _ldb, _C_Host, _ldc, _alpha, _beta);
        t_h.end();
        cout << "cpu elapsed time : " << t_h.elapsed() << " ms,  GFLOPS: " << gflops(2 * _m * _n * _k, t_h.elapsed()) << endl;
}

template class Gemm<float>;
//template class Gemm<double>;