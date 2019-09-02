#ifndef COMMON_H
#define COMMON_H

#include"config.h"
#include<iostream>

#define cord(r, c, ld) ((ld) * (c) + (r))

template<typename DataType>
void print(DataType* m, int rows, int cols);

template<typename DataType>
void fillRandom(DataType* m, int size);
float gflops(long long ins_num, float t);


#define errCheck(stat1, stat2) { errCheck_(stat1, stat2, __FILE__, __LINE__); }
inline void errCheck_(bool stat1, bool stat2, const char* file, int line){
    if(stat1 != stat2){
        std::cerr << "Err Occured! " << file << " " << line << std::endl;
        exit(1);
    }
}

#endif