#include "gemm.h"
#include "timer.h"
#include "impl/cpu/gemm_cpu.h"

#ifdef USE_GPU
#include "impl/gpu/gemm_gpu.h"
#endif

using namespace std;
template<typename DataType>
void Gemm<DataType>::gpu(std::vector<float>& t, int warm_up, int iter_num){
#ifdef USE_GPU
    gemm_gpu(_m, _n, _k, _A, _lda, _B, _ldb, _C, _ldc, _alpha, _beta, &_C_Dev_Host, t, warm_up, iter_num);
#endif
}

template<typename DataType>
void Gemm<DataType>::cpu(std::vector<float>& t, int warm_up, int iter_num){
        int c_size = _m * _n;
        _C_Host = static_cast<DataType*>(malloc(c_size * sizeof(DataType)));
        
        memcpy(_C_Host, _C, c_size * sizeof(float));


        // warm up
        for(int i = 0; i  < warm_up; ++i){
            gemm_cpu(_m, _n, _k, _A, _lda, _B, _ldb, _C_Host, _ldc, _alpha, _beta);    
        }

        Timer<CPU> t_h;

        for(int i = 0; i < iter_num; ++i){
            t_h.start();
            gemm_cpu(_m, _n, _k, _A, _lda, _B, _ldb, _C_Host, _ldc, _alpha, _beta);
            t_h.end();    
        }
        
        cout << "cpu elapsed time : " << t_h.getAverageTimeMs() << " ms,  GFLOPS: " << gflops(2 * _m * _n * _k, t_h.getAverageTimeMs()) << endl;
        t.emplace_back(t_h.getAverageTimeMs());
}

template class Gemm<float>;
//template class Gemm<double>;