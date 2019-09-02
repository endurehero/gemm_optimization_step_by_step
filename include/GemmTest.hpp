#ifndef INCLUDE_GEMM_TEST_HPP
#define INCLUDE_GEMM_TEST_HPP

#include <vector>
#include <cmath>
#include <string.h>
#include "GemmFactory.hpp"
#include "recorder.hpp"
#include "GemmTestCase.hpp"



class GemmTest{
public:
    GemmTest(int warm_up = 0, int iter_num = 1, Recorder* timer = nullptr)
        :_warm_up(warm_up),
         _iter_num(iter_num),
         _time_recorder(timer){}

    template<typename T>
    void perf(GEMM_ALGO_TYPE type, const std::vector<GemmTestCase<T>>& test_case){
        auto algo_ptr = GemmFactory<T>::getGemmAlgo(type);
        auto& algo = *algo_ptr;

        algo.setRunTime(_warm_up, _iter_num);
        
        printf("Perf algo: %s, test case num: %d, warm_up = %d, iter_num = %d\n", algoType2Name(type).c_str(), test_case.size(), _warm_up, _iter_num);
        
        int m, n, k, lda, ldb, ldc;
        T alpha, beta;
        for(int i = 0; i < test_case.size(); ++i){
            printf("[Case %d]", i);
            m = test_case[i].m;
            n = test_case[i].n;
            k = test_case[i].k;
            lda = test_case[i].lda;
            ldb = test_case[i].ldb;
            ldc = test_case[i].ldc;
            alpha = test_case[i].alpha;
            beta = test_case[i].beta;
            
            int size_a = lda * k, size_b = ldb * n, size_c = ldc * n;
            T* a = (T*)malloc(size_a * sizeof(T));
            T* b = (T*)malloc(size_b * sizeof(T));
            T* c = (T*)malloc(size_c * sizeof(T));

            fillRandom(a, size_a);
            fillRandom(b, size_b);
            fillRandom(c, size_c);

            algo(false, false, m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);


            printf("m = %d, n = %d, k = %d, lda = %d, ldb = %d, ldc = %d, alpha = %f, beta = %f",\
                m, n, k, lda, ldb, ldc, alpha, beta);
            printf(",elapsed time: %f ms.\n", algo.elapsed());

            if(nullptr != _time_recorder){
                *(_time_recorder) << algoType2Name(type) << " " << m << " " << n << " " << k << " " << lda << " " << ldb << " " << ldc << " " << alpha << " " << beta << " " << algo.elapsed() << "\n";
            }

            free(a); free(b); free(c);
                
        }

        printf("Perf completed! algo: %s, test case num: %d, warm_up = %d, iter_num = %d\n", algoType2Name(type).c_str(), test_case.size(), _warm_up, _iter_num);
    }


    template<typename T>
    void verify(GEMM_ALGO_TYPE type1, GEMM_ALGO_TYPE type2, const std::vector<GemmTestCase<T>>& test_case){
        auto algo1_ptr = GemmFactory<T>::getGemmAlgo(type1);
        auto algo2_ptr = GemmFactory<T>::getGemmAlgo(type2);

        auto& algo1 = *algo1_ptr;
        auto& algo2 = *algo2_ptr;

        printf("Verify 2 algos, algo1: %s, algo2: %s. case num: %d\n", algoType2Name(type1).c_str(), algoType2Name(type2).c_str(), test_case.size());

        int m, n, k, lda, ldb, ldc;
        T alpha, beta;
        for(int i = 0; i < test_case.size(); ++i){
            printf("[Case %d]", i);

            m = test_case[i].m;
            n = test_case[i].n;
            k = test_case[i].k;
            lda = test_case[i].lda;
            ldb = test_case[i].ldb;
            ldc = test_case[i].ldc;
            alpha = test_case[i].alpha;
            beta = test_case[i].beta;
            
            int size_a = lda * k, size_b = ldb * n, size_c = ldc * n;
            T* a = (T*)malloc(size_a * sizeof(T));
            T* b = (T*)malloc(size_b * sizeof(T));
            T* c1 = (T*)malloc(size_c * sizeof(T));
            T* c2 = (T*)malloc(size_c * sizeof(T));
            

            fillRandom(a, size_a);
            fillRandom(b, size_b);
            fillRandom(c1, size_c);
            memcpy(c2, c1, sizeof(T) * size_c);

            algo1(false, false, m, n, k, a, lda, b, ldb, c1, ldc, alpha, beta);
            algo2(false, false, m, n, k, a, lda, b, ldb, c2, ldc, alpha, beta);

            T max_diff, max_ratio;
            cmp<T>(c1, c2, size_c, max_diff, max_ratio);

            if(max_diff <= _diff_threshold && max_ratio <= _ratio_threshold){
                //printf(" [%s agreed with %s] ", algoType2Name(type1).c_str(), algoType2Name(type2).c_str());
                printf(" [PASSED] ");
            }
            else{
                //printf(" [%s disagreed with %s] ", algoType2Name(type1).c_str(), algoType2Name(type2).c_str());
                printf(" [FAILED] ");
            }
            
            printf("m = %d, n = %d, k = %d, lda = %d, ldb = %d, ldc = %d, alpha = %f, beta = %f",\
                m, n, k, lda, ldb, ldc, alpha, beta);
            printf(", max_diff = %f, max_ratio = %f\n", max_diff, max_ratio);
            
            

            free(a); free(b); free(c1); free(c2);
        }

    }

    
    void setThreshold(float diff_threshold, float ratio_threshold){
        _diff_threshold = diff_threshold;
        _ratio_threshold = ratio_threshold;
    }

    float diffThreshold() const { return _diff_threshold; }
    float ratioThreshold() const { return _ratio_threshold; }

private:
    int _warm_up;
    int _iter_num;
    Recorder* _time_recorder;

    float _diff_threshold{0.01};
    float _ratio_threshold{0.00001};

    const float _eps{0.0000001};

    template<typename T>
    void cmp(T* a, T* b, int size, T& max_diff, T& max_ratio){
        ErrCheckExt(nullptr == a || nullptr == b, false);
        
        max_diff = fabs(a[0] - b[0]);
        max_ratio = 2 * max_diff / (a[0] + b[0] + _eps);

        for(int i = 1; i < size; ++i){
            T diff = fabs(a[i] - b[i]);
            if(diff > max_diff){
                max_diff = diff;
                max_ratio = 2 * max_diff / (a[i] + b[i] + _eps);
            }
        }
    }
    
    
};
#endif // INCLUDE_GEMM_TEST_HPP