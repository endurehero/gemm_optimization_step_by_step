#ifndef INCLUDE_GEMM_BASE_HPP
#define INCLUDE_GEMM_BASE_HPP
#include "common.h"
#include "timer.hpp"

template<typename T>
class GemmBase{
public:
    GemmBase(int warm_up = 0, int iter_num = 1)
        :_warm_up(warm_up),
         _iter_num(iter_num){}
         
    virtual ~GemmBase(){}
    
    virtual void operator()(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta) = 0;

    int warp() const { return _warm_up; }
    int iter() const { return _iter_num; }
    float elapsed() const { return _elapsed; }

    void setRunTime(int warm_up = 0, int iter_num = 0){
        _warm_up = warm_up;
        _iter_num = iter_num;
    }

protected:
    int _warm_up;
    int _iter_num;
    float _elapsed{0.0};

    virtual void gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta) = 0;

};

#endif // INCLUDE_GEMM_BASE_HPP