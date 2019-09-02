#include "include/common.h"
#include <iostream>
#include <random>
using namespace std;


unordered_map<int, std::string> _algoType2Name{
    {CPU_RAW, "CPU_RAW"}, {CPU_CBLAS, "CPU_CBLAS"}, {CPU_OPT1, "CPU_OPT1"},
    {CPU_OPT2, "CPU_OPT2"}, {CPU_OPT3, "CPU_OPT3"}, {CPU_OPT4, "CPU_OPT4"},
    {CPU_OPT5, "CPU_OPT5"}, {CPU_OPT6, "CPU_OPT6"}, {CPU_OPT7, "CPU_OPT7"},
    {CPU_OPT6, "CPU_OPT6"}, {CPU_OPT7, "CPU_OPT7"}, {CPU_OPT8, "CPU_OPT8"},
    {CPU_OPT9, "CPU_OPT9"},
    
    {GPU_RAW, "GPU_RAW"}, {GPU_CUBLAS, "GPU_CUBLAS"}, {GPU_CUBLAS_TENSORCORE, "GPU_CUBLAS_TENSORCORE"},
    {GPU_OPT1, "GPU_OPT1"}, {GPU_OPT2, "GPU_OPT2"}, {GPU_OPT3, "GPU_OPT3"}
    };

unordered_map<std::string, int> _algoName2Type{
    {"CPU_RAW", CPU_RAW}, {"CPU_CBLAS", CPU_CBLAS}, {"CPU_OPT1", CPU_OPT1},
    {"CPU_OPT2", CPU_OPT2}, {"CPU_OPT3", CPU_OPT3}, {"CPU_OPT4", CPU_OPT4},
    {"CPU_OPT5", CPU_OPT5}, {"CPU_OPT6", CPU_OPT6}, {"CPU_OPT7", CPU_OPT7},
    {"CPU_OPT6", CPU_OPT6}, {"CPU_OPT7", CPU_OPT7}, {"CPU_OPT8", CPU_OPT8},
    {"CPU_OPT9", CPU_OPT9},
    
    {"GPU_RAW", GPU_RAW}, {"GPU_CUBLAS", GPU_CUBLAS}, {"GPU_CUBLAS_TENSORCORE", GPU_CUBLAS_TENSORCORE},
    {"GPU_OPT1", GPU_OPT1}, {"GPU_OPT2", GPU_OPT2}, {"GPU_OPT3", GPU_OPT3}
};

GEMM_ALGO_TYPE algoName2Type(std::string name){
    return static_cast<GEMM_ALGO_TYPE>(_algoName2Type[name]);
}
std::string algoType2Name(GEMM_ALGO_TYPE type){
    return _algoType2Name[static_cast<int>(type)];
}


template<typename DataType>
void fillRandom(DataType* m, int size){
    default_random_engine e;
    uniform_real_distribution<DataType> u(-10, 10);
    
    for(int i = 0; i < size; ++i){
        m[i] = u(e);
    }
}


float gflops(long long ins_num, float t){
        return static_cast<float>(static_cast<float>(ins_num) / t / 1000000);
}

float efficiency(float self, float target){
    return self / target;
}

template void fillRandom<float>(float*, int);
template void fillRandom<double>(double*, int);