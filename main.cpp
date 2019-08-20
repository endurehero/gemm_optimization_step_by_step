#include "gemm.h"
#include "common.h"

int main(){
    int m = 1000;
    int n = 1000;
    int k = 256;
    int size_a = m * k;
    int size_b = n * k;
    int size_c = m * n;
    float alpha = 1.0, beta = 0.0;
    
    double* A = static_cast<double*>(malloc(size_a * sizeof(double)));
    double* B = static_cast<double*>(malloc(size_b * sizeof(double)));
    double* C = static_cast<double*>(malloc(size_c * sizeof(double)));

    fillRandom(A, size_a);
    fillRandom(B, size_b);
    fillRandom(C, size_c);

    Gemm<double> gemm(false, false, m, n, k, A, m, B, k, C, m, alpha, beta);
    
    
    gemm.cpu();
    
#ifdef USE_GPU
    gemm.gpu();
    float max_diff = 0.0, max_ratio = 0.0;
    gemm.cmp(max_diff, max_ratio);

    cout << "max_diff = " << max_diff << " max_ratio = " << max_ratio << endl;
#endif
    
#if 0
    cout << "A:" << endl;
    print(A, m, k);
    
    cout << "B:" << endl;
    print(B, k, n);
    
    cout << "Host:" << endl;
    print(gemm.c_host(), m, n);

    cout << "Dev: " << endl;
    print(gemm.c_dev_host(), m, n);
#endif


    free(A); A = nullptr;
    free(B); B = nullptr;
    free(C); C = nullptr;

    return 0;
}