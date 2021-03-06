#if 0

#include "impl/cpu/gemm_cpu.h"

#ifdef USE_CXX11_THREAD
#include<thread>
#include<algorithm>

using namespace std;

/**
 * / brief implement gemm with Block4X4 and unroll the col loop and more register.
 */
#define UNROLL_SIZE 4

typedef struct{
    int m;
    int n;
    int k;
    float* a;
    int lda;
    float* b;
    int ldb;
    float* c;
    int ldc;
    float alpha;
    float beta;
}thread_param_t;


#if 0
void addDot4X4Reg(int k, float* a, int lda, float*b, int ldb, float* c, int ldc, float alpha, float beta){
    
    register float 
        c_00_reg = 0.0, c_01_reg = 0.0, c_02_reg = 0.0, c_03_reg = 0.0,
        c_10_reg = 0.0, c_11_reg = 0.0, c_12_reg = 0.0, c_13_reg = 0.0,
        c_20_reg = 0.0, c_21_reg = 0.0, c_22_reg = 0.0, c_23_reg = 0.0,
        c_30_reg = 0.0, c_31_reg = 0.0, c_32_reg = 0.0, c_33_reg = 0.0;
        
    register float 
        a_p0_reg = 0.0,
        a_p1_reg = 0.0,
        a_p2_reg = 0.0,
        a_p3_reg = 0.0,
        b_0p_reg = 0.0,
        b_1p_reg = 0.0,
        b_2p_reg = 0.0,
        b_3p_reg = 0.0;

    float
        *a_p0_ptr = a,
        *a_p1_ptr = a + 1,
        *a_p2_ptr = a + 2,
        *a_p3_ptr = a + 3,
        *b_0p_ptr = b,
        *b_1p_ptr = b + ldb,
        *b_2p_ptr = b + 2 * ldb,
        *b_3p_ptr = b + 3 * ldb;
        

    for(int p = 0; p < k; ++p){
        b_0p_reg = *b_0p_ptr++;
        b_1p_reg = *b_1p_ptr++;
        b_2p_reg = *b_2p_ptr++;
        b_3p_reg = *b_3p_ptr++;

        // first row
        a_p0_reg = a_p0_ptr[p * lda];
        c_00_reg += a_p0_reg * b_0p_reg;
        c_01_reg += a_p0_reg * b_1p_reg;
        c_02_reg += a_p0_reg * b_2p_reg;
        c_03_reg += a_p0_reg * b_3p_reg;

        // second row
        a_p1_reg = a_p1_ptr[p * lda];
        c_10_reg += a_p1_reg * b_0p_reg;
        c_11_reg += a_p1_reg * b_1p_reg;
        c_12_reg += a_p1_reg * b_2p_reg;
        c_13_reg += a_p1_reg * b_3p_reg;
        
        // third row
        a_p2_reg = a_p2_ptr[p * lda];
        c_20_reg += a_p2_reg * b_0p_reg;
        c_21_reg += a_p2_reg * b_1p_reg;
        c_22_reg += a_p2_reg * b_2p_reg;
        c_23_reg += a_p2_reg * b_3p_reg;

        // fouth row
        a_p3_reg = a_p3_ptr[p * lda];
        c_30_reg += a_p3_reg * b_0p_reg;
        c_31_reg += a_p3_reg * b_1p_reg;
        c_32_reg += a_p3_reg * b_2p_reg;
        c_33_reg += a_p3_reg * b_3p_reg;
    }

    // first row
    c[cord(0, 0, ldc)] = alpha * c_00_reg + beta * c[cord(0, 0, ldc)];
    c[cord(0, 1, ldc)] = alpha * c_01_reg + beta * c[cord(0, 1, ldc)];
    c[cord(0, 2, ldc)] = alpha * c_02_reg + beta * c[cord(0, 2, ldc)];
    c[cord(0, 3, ldc)] = alpha * c_03_reg + beta * c[cord(0, 3, ldc)];
    // second row
    c[cord(1, 0, ldc)] = alpha * c_10_reg + beta * c[cord(1, 0, ldc)];
    c[cord(1, 1, ldc)] = alpha * c_11_reg + beta * c[cord(1, 1, ldc)];
    c[cord(1, 2, ldc)] = alpha * c_12_reg + beta * c[cord(1, 2, ldc)];
    c[cord(1, 3, ldc)] = alpha * c_13_reg + beta * c[cord(1, 3, ldc)];
    // third row
    c[cord(2, 0, ldc)] = alpha * c_20_reg + beta * c[cord(2, 0, ldc)];
    c[cord(2, 1, ldc)] = alpha * c_21_reg + beta * c[cord(2, 1, ldc)];
    c[cord(2, 2, ldc)] = alpha * c_22_reg + beta * c[cord(2, 2, ldc)];
    c[cord(2, 3, ldc)] = alpha * c_23_reg + beta * c[cord(2, 3, ldc)];
    // fouth row
    c[cord(3, 0, ldc)] = alpha * c_30_reg + beta * c[cord(3, 0, ldc)];
    c[cord(3, 1, ldc)] = alpha * c_31_reg + beta * c[cord(3, 1, ldc)];
    c[cord(3, 2, ldc)] = alpha * c_32_reg + beta * c[cord(3, 2, ldc)];
    c[cord(3, 3, ldc)] = alpha * c_33_reg + beta * c[cord(3, 3, ldc)];
}

//void gemm_cpu_per_thread(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta){
void gemm_cpu_per_thread(const thread_param_t& param){
    int m = param.m, n = param.n, k = param.k;
    float* a = param.a; int lda = param.lda;
    float* b = param.b; int ldb = param.ldb;
    float* c = param.c; int ldc = param.ldc;
    float alpha = param.alpha, beta = param.beta;
    
    for(int col = 0; col < n; col += UNROLL_SIZE){
        for(int row = 0; row < m; row += UNROLL_SIZE){
            
            float* a_head = &(a[cord(row, 0, lda)]);
            float* b_head = &(b[cord(0, col, ldb)]);
            float* c_head = &(c[cord(row, col, ldc)]);
            addDot4X4Reg(k, a_head, lda, b_head, ldb, c_head, ldc, alpha, beta);
        }
    }
    
}

#endif



void gemm_cpu(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta){

    if(m % 4 != 0 || n % 4 != 0 || 0 == m || 0 == n){
        cout << "Optimize8: Param m = " << m << " n = " << n << " not implemented!" << endl;
        return;
    }
    
    int max_phy_thread_num = thread::hardware_concurrency();
    bool is_row_split = true; if(n > m) is_row_split = false;
    int m_block4 = m / UNROLL_SIZE, n_block4 = n / UNROLL_SIZE;
    int real_thread_num = min(max_phy_thread_num, (is_row_split) ? m_block4 : n_block4);
    real_thread_num = 1;

    //cout << "max_phy_thread_num = " << max_phy_thread_num << " real_thread_num = " << real_thread_num << endl;


    thread_param_t* params = new thread_param_t[real_thread_num]; 
    
    int offset = (is_row_split) ? (m_block4 / real_thread_num * UNROLL_SIZE) : (n_block4 / real_thread_num * UNROLL_SIZE);
    for(int i = 0; i < real_thread_num; ++i){
        if(is_row_split){
            params[i].m = min(offset, m - i * offset);
            params[i].n = n;
            params[i].a = &a[cord(i * offset, 0, lda)];
            params[i].b = b;
            params[i].c = &c[cord(i * offset, 0, ldc)];
        }
        else{
            params[i].m = m;
            params[i].n = min(i * offset, n - i * offset);
            params[i].a = a;
            params[i].b = &b[cord(0, i * offset, ldb)];
            params[i].c = &c[cord(0, i * offset, ldc)];
        }

        params[i].k = k;
        params[i].lda = lda;
        params[i].ldb = ldb;
        params[i].ldc = ldc;
        params[i].alpha = alpha;
        params[i].beta = beta;
    }

    thread* threads = new thread[real_thread_num];
    for(int i = 0; i < real_thread_num; ++i){
        threads[i] = thread(gemm_cpu_per_thread, params[i]);
    }

    for(int i = 0; i < real_thread_num; ++i){
        threads[i].join();
    }


    delete [] threads;
    delete [] params;
    
}


#endif

#endif