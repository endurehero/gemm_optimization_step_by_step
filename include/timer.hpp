#ifndef INCLUDE_TIMER_HPP
#define INCLUDE_TIMER_HPP

#include <chrono>
#include <list>

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif


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
        _start = std::chrono::system_clock::now();
    }
    
    void end(){
        _end = std::chrono::system_clock::now();
        _time_record.push_back(elapsed());
    }

    float elapsed(){
        return std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
    }
    
    void clear(){
        _time_record.clear();
    }

    float getAverageTimeMs(){
        float acc = 0.0;
        for(auto itr = _time_record.begin(); itr != _time_record.end(); ++itr){
            acc += *itr;
        }

        acc /= _time_record.size();
        return acc;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> _start;
    std::chrono::time_point<std::chrono::system_clock> _end;
    std::list<float> _time_record;

};

#ifdef USE_GPU
template<>
class Timer<NV>{
public:
    Timer(cudaStream_t stream = 0)
        :_stream(stream){
        cudaEventCreate(&_start);
        cudaEventCreate(&_end);
    }

    ~Timer(){
        cudaEventDestroy(_start);
        cudaEventDestroy(_end);
    }

    void start(){
        cudaEventRecord(_start, _stream);
    }
    
    void end(){
        cudaEventRecord(_end, _stream);
        cudaEventSynchronize(_end);

        _time_record.push_back(elapsed());
    }

    float elapsed(){
        float t;
        cudaEventElapsedTime(&t, _start, _end);
        return t;
    }

    void clear(){
        _time_record.clear();
    }

    float getAverageTimeMs(){
        float acc = 0.0;
        for(auto itr = _time_record.begin(); itr != _time_record.end(); ++itr){
            acc += *itr;
        }

        acc /= _time_record.size();
        return acc;
    }

private:
    cudaStream_t _stream;
    cudaEvent_t _start, _end;
    std::list<float> _time_record;

};
#endif // USE_GPU
#endif // INCLUDE_TIMER_HPP