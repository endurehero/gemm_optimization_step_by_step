#include <iostream>
#include <random>
using namespace std;


void print(float* m, int rows, int cols){
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

void fillRandom(float* m, int size){
    default_random_engine e;
    uniform_real_distribution<float> u(-10, 10);
    
    for(int i = 0; i < size; ++i){
        m[i] = u(e);
    }
}
