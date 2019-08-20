#include <iostream>
#include <random>
using namespace std;

template<typename DataType>
void print(DataType* m, int rows, int cols){
    if(nullptr == m){
        cout << "Invalid matrix" << endl;
        return;
    }
    
    for(int r = 0; r < rows; ++r){
        for(int c = 0; c < cols; ++c){
            cout << m[r * cols + c] << " ";
        }

        cout << endl;
    }
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