#ifndef INCLUDE_GEMM_TEST_CASE_HPP
#define INCLUDE_GEMM_TEST_CASE_HPP

template<typename T>
struct GemmTestCase{
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    T alpha;
    T beta;
    
    GemmTestCase() = default;
    GemmTestCase(int M, int N, int K, int LDA, int LDB, int LDC, T ALPHA, T BETA)
        :m(M), n(N), k(K), lda(LDA), ldb(LDB), ldc(LDC), alpha(ALPHA), beta(BETA){}

};

#endif //INCLUDE_GEMM_TEST_CASE_HPP