#include "gemm.h"
#include "common.h"
#include "record.hpp"
#include <vector>

Recorder record;
int warp_up = 0;
int iter_num = 1;

void test(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta){

    std::cout << "********************************************************************" << std::endl;
    std::cout << "TEST: m = " << m << \
                 " n = " << n << \
                 " k = " << k << \
                 " lda = " << lda << \
                 " ldb = " << ldb << \
                 " ldc = " << ldc << \
                 " alpha = " << alpha << \
                 " beta = " << beta << \
                 " warm up = " << warp_up << \
                 " iter num = " << iter_num << std::endl;
 
    int size_a = m * k;
    int size_b = n * k;
    int size_c = m * n;
    
    float* A = static_cast<float*>(malloc(size_a * sizeof(float)));
    float* B = static_cast<float*>(malloc(size_b * sizeof(float)));
    float* C = static_cast<float*>(malloc(size_c * sizeof(float)));

    fillRandom<float>(A, size_a);
    fillRandom<float>(B, size_b);
    fillRandom<float>(C, size_c);

    Gemm<float> gemm(false, false, m, n, k, A, m, B, k, C, m, alpha, beta);
    
    std::vector<float> t_cpu, t_gpu;
    
    gemm.cpu(t_cpu, warp_up, iter_num);
    
#ifdef USE_GPU
    gemm.gpu(t_gpu, warp_up, iter_num);
    float max_diff = 0.0, max_ratio = 0.0;
    gemm.cmp(max_diff, max_ratio);

    std::cout << "max_diff = " << max_diff << " max_ratio = " << max_ratio << std::endl;
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


    record << m << " " << n << " " << k << " " << lda << " " << " " << ldb << " " << ldc << " " << alpha << " " << beta;
    // record cpu time.
    for(auto t_c : t_cpu) record << " " << t_c;
    // reocrd gpu time
    for(auto t_g : t_gpu) record << " " << t_g;
    
    record << "\n";

    std::cout << "********************************************************************" << std::endl;
}

int main(int argc, char** argv){

    if(argc >= 2){
        if("1" == argv[1]){
            warp_up = 3;
            iter_num = 100;
        }
    }

    if(argc >= 3){
        std::cout << "perf data will be store at " << argv[2] << std::endl;
        record.setPath(argv[2]);
    }

    

    // small size
    for(int size = 64; size < 1024; size += 128){
        test(size, size, size, size, size, size, 1.0, 1.0);
    }

    // big size
    for(int size = 1024; size <= 4096; size += 512){
        test(size, size, size, size, size, size, 1.0, 1.0);
    }

    return 0;
}