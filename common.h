#ifndef COMMON_H
#define COMMON_H

#include"config.h"
#include<iostream>
using namespace std;

#define cord(r, c, ld) ((ld) * (c) + (r))

template<typename DataType>
void print(DataType* m, int rows, int cols);

template<typename DataType>
void fillRandom(DataType* m, int size);
float gflops(long long ins_num, float t);


#endif