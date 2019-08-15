#ifndef COMMON_H
#define COMMON_H

#include<iostream>
using namespace std;


#define cord(r, c, ld) ((ld) * (c) + (r))



void print(float* m, int rows, int cols);
void fillRandom(float* m, int size);
#endif