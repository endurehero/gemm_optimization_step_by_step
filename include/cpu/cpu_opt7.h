#ifndef INCLUDE_CPU_CPU_OPT7_H
#define INCLUDE_CPU_CPU_OPT7_H
#include "include/GemmFactory.hpp"

template<typename T>
class CpuOpt7 : public GemmBase<T>{
public:
    typedef GemmBase<T> Base;


    CpuOpt7(int warm_up = 0, int iter_num = 1)
        :GemmBase<T>(warm_up, iter_num){}

    virtual void operator()(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta) override;

private:
    virtual void gemm(bool transA, bool transB, int m, int n, int k, \
        T* a, int lda, T* b, int ldb, T* c, int ldc, T alpha, T beta) override;
};

#endif //INCLUDE_CPU_CPU_OPT7_H