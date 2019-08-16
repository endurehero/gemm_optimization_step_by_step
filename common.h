#ifndef COMMON_H
#define COMMON_H

#include"config.h"
#include<iostream>
using namespace std;

#define cord(r, c, ld) ((ld) * (c) + (r))

void print(float* m, int rows, int cols);
void fillRandom(float* m, int size);
float gflops(long long ins_num, float t);


#endif