#ifndef GEMM_H
#define GEMM_H

#include <memory.h>
#include <stdlib.h>
#include <cmath>

template<typename DataType>
class Gemm{
public:
    Gemm(bool transA, bool transB, int m, int n, int k,
        DataType* A, int lda, DataType* B, int ldb, DataType* C, int ldc,
        DataType alpha, DataType beta)
        :_transA(transA),
         _transB(transB),
         _m(m), _n(n), _k(k),
         _lda(lda), _ldb(ldb),
         _ldc(ldc), _alpha(alpha),_beta(beta),
         _A(A), _B(B), _C(C){}
    
    ~Gemm(){
        
        if(nullptr != _C_Host){
            delete [] _C_Host;
            _C_Host = nullptr;
        }
        if(nullptr != _C_Dev_Host){
            delete [] _C_Dev_Host;
            _C_Dev_Host = nullptr;
        }
    }

    void cpu();
    void gpu();

    void cmp(DataType& max_diff, DataType& max_ratio){
        DataType* h = _C_Host;
        DataType* d = _C_Dev_Host;
        int size = _m * _n;
        
        max_diff = fabs(h[0] - d[0]);
        max_ratio = 2 * max_diff / (h[0] + d[0] + eps);

        for(int i = 1; i < size; ++i){
            DataType diff = fabs(h[i] - d[i]);
            if(diff > max_diff){
                max_diff = diff;
                max_ratio = 2 * max_diff / (h[i] + d[i] + eps);
            }
        }
    }

    DataType* c_host(){ return _C_Host; }
    DataType* c_dev_host(){ return _C_Dev_Host; }

private:
    const float eps{0.000001};

    bool _transA{false};
    bool _transB{false};
    int _m;
    int _n;
    int _k;
    int _lda;
    int _ldb;
    int _ldc;
    DataType _alpha;
    DataType _beta;
    DataType* _A{nullptr};
    DataType* _B{nullptr};
    DataType* _C{nullptr};

    DataType* _C_Host{nullptr};
    DataType* _C_Dev_Host{nullptr};
};

#endif