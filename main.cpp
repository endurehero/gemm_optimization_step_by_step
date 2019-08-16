#include "timer.h"
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
    
    float* A = static_cast<float*>(malloc(size_a * sizeof(float)));
    float* B = static_cast<float*>(malloc(size_b * sizeof(float)));
    float* C = static_cast<float*>(malloc(size_c * sizeof(float)));

    fillRandom(A, size_a);
    fillRandom(B, size_b);
    fillRandom(C, size_c);

    Gemm gemm(false, false, m, n, k, A, m, B, k, C, m, alpha, beta);
    
    Timer<CPU> t_h;
    t_h.start();
    gemm.cpu();
    t_h.end();
    cout << "cpu elapsed time : " << t_h.elapsed() << " ms,  GFLOPS: " << gflops(2 * m * n * k, t_h.elapsed()) << endl;
    
#ifdef USE_GPU
    gemm.gpu();
    float max_diff = 0.0, max_ratio = 0.0;
    gemm.cmp(max_diff, max_ratio);

    cout << "max_diff = " << max_diff << " max_ratio = " << max_ratio << endl;
#endif
    
#ifdef ENABLE_DEBUG
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