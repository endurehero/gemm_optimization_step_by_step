#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

struct NV{};
struct CPU{};

template <typename TargetType>
class Timer{
public:
    Timer(){
    }

    ~Timer(){
    }

    void start(){
        _start = system_clock::now();
    }
    
    void end(){
        _end = system_clock::now();
    }

    float elapsed(){
        return duration_cast<milliseconds>(_end - _start).count();
    }

    long long gflops(long long ins_num, float t){
        return static_cast<long long>(ins_num / t / 1000000);
    }
    

private:
    time_point<system_clock> _start;
    time_point<system_clock> _end;

};

template<>
class Timer<NV>{
public:
    Timer(){
        cudaEventCreate(&_start);
        cudaEventCreate(&_end);
    }

    ~Timer(){
        cudaEventDestroy(_start);
        cudaEventDestroy(_end);
    }

    void start(){
        cudaEventRecord(_start);
    }
    
    void end(){
        cudaEventRecord(_end);
        cudaEventSynchronize(_end);
    }

    float elapsed(){
        float t;
        cudaEventElapsedTime(&t, _start, _end);
        return t;
    }

    long long gflops(long long ins_num, float t){
        return static_cast<long long>(ins_num / t / 1000000);
    }
    

private:
    cudaEvent_t _start, _end;

};

#endif