
#include "impl/cpu/gemm_cpu.h"

#ifdef USE_CPU_OPT9
#include <mmintrin.h>
#include <xmmintrin.h> // SSE
#include <pmmintrin.h> // SSE2
#include <emmintrin.h> // SSE3
#include <algorithm>
#include <omp.h>
/**
 * / brief implement gemm optimized with.
 *         1. unroll 4 loop
 *         2. decrease addressing overhead
 *         3. use sse intrin(128 bits vreg)
 *         4. cache the macro tile 128*128 into L2 cache(256KB)
 */
using namespace std;

typedef union{
    __m128 v;
    float d[4];
}v4f_t;

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

#define m_block 64
#define n_block 64

void addDot4X4VReg(int k, float* a, int lda, float*b, int ldb, float* c, int ldc, float alpha, float beta){
    
    v4f_t
        c_col0_vreg, c_col1_vreg, c_col2_vreg, c_col3_vreg,
        a_p_vreg, b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

    float
        *b_p0_ptr = b,
        *b_p1_ptr = b + ldb,
        *b_p2_ptr = b + 2 * ldb,
        *b_p3_ptr = b + 3 * ldb;
    
    c_col0_vreg.v = _mm_setzero_ps();
    c_col1_vreg.v = _mm_setzero_ps();
    c_col2_vreg.v = _mm_setzero_ps();
    c_col3_vreg.v = _mm_setzero_ps();

    for(int p = 0; p < k; ++p){
        a_p_vreg.v = _mm_load_ps(static_cast<float*>(a)); a += lda;
        b_p0_vreg.v = _mm_load1_ps(static_cast<float*>(b_p0_ptr++));
        b_p1_vreg.v = _mm_load1_ps(static_cast<float*>(b_p1_ptr++));
        b_p2_vreg.v = _mm_load1_ps(static_cast<float*>(b_p2_ptr++));
        b_p3_vreg.v = _mm_load1_ps(static_cast<float*>(b_p3_ptr++));

        c_col0_vreg.v += a_p_vreg.v * b_p0_vreg.v;
        c_col1_vreg.v += a_p_vreg.v * b_p1_vreg.v;
        c_col2_vreg.v += a_p_vreg.v * b_p2_vreg.v;
        c_col3_vreg.v += a_p_vreg.v * b_p3_vreg.v;
    }

    // first col
    c[cord(0, 0, ldc)] = alpha * c_col0_vreg.d[0] + beta * c[cord(0, 0, ldc)];
    c[cord(1, 0, ldc)] = alpha * c_col0_vreg.d[1] + beta * c[cord(1, 0, ldc)];
    c[cord(2, 0, ldc)] = alpha * c_col0_vreg.d[2] + beta * c[cord(2, 0, ldc)];
    c[cord(3, 0, ldc)] = alpha * c_col0_vreg.d[3] + beta * c[cord(3, 0, ldc)];
    // second col
    c[cord(0, 1, ldc)] = alpha * c_col1_vreg.d[0] + beta * c[cord(0, 1, ldc)];
    c[cord(1, 1, ldc)] = alpha * c_col1_vreg.d[1] + beta * c[cord(1, 1, ldc)];
    c[cord(2, 1, ldc)] = alpha * c_col1_vreg.d[2] + beta * c[cord(2, 1, ldc)];
    c[cord(3, 1, ldc)] = alpha * c_col1_vreg.d[3] + beta * c[cord(3, 1, ldc)];
    // third col
    c[cord(0, 2, ldc)] = alpha * c_col2_vreg.d[0] + beta * c[cord(0, 2, ldc)];
    c[cord(1, 2, ldc)] = alpha * c_col2_vreg.d[1] + beta * c[cord(1, 2, ldc)];
    c[cord(2, 2, ldc)] = alpha * c_col2_vreg.d[2] + beta * c[cord(2, 2, ldc)];
    c[cord(3, 2, ldc)] = alpha * c_col2_vreg.d[3] + beta * c[cord(3, 2, ldc)];
    // fouth
    c[cord(0, 3, ldc)] = alpha * c_col3_vreg.d[0] + beta * c[cord(0, 3, ldc)];
    c[cord(1, 3, ldc)] = alpha * c_col3_vreg.d[1] + beta * c[cord(1, 3, ldc)];
    c[cord(2, 3, ldc)] = alpha * c_col3_vreg.d[2] + beta * c[cord(2, 3, ldc)];
    c[cord(3, 3, ldc)] = alpha * c_col3_vreg.d[3] + beta * c[cord(3, 3, ldc)];
}

void pack_a(int k, float* src_a, int src_lda, float* dst_a, int dst_lda){
    float* src_r0_a = src_a,     *dst_r0_a = dst_a;
    float* src_r1_a = src_a + 1, *dst_r1_a = dst_a + 1;
    float* src_r2_a = src_a + 2, *dst_r2_a = dst_a + 2;
    float* src_r3_a = src_a + 3, *dst_r3_a = dst_a + 3;

    for(int p = 0; p < k; ++p){
        *dst_r0_a = *src_r0_a; src_r0_a += src_lda; dst_r0_a += dst_lda;
        *dst_r1_a = *src_r1_a; src_r1_a += src_lda; dst_r1_a += dst_lda;
        *dst_r2_a = *src_r2_a; src_r2_a += src_lda; dst_r2_a += dst_lda;
        *dst_r3_a = *src_r3_a; src_r3_a += src_lda; dst_r3_a += dst_lda;
    }
}

void pack_b(int k, float* src_b, int src_ldb, float* dst_b, int dst_ldb){
    float* src_c0_b = src_b,               *dst_c0_b = dst_b;
    float* src_c1_b = src_b + src_ldb,     *dst_c1_b = dst_b + dst_ldb;
    float* src_c2_b = src_b + 2 * src_ldb, *dst_c2_b = dst_b + 2 * dst_ldb;
    float* src_c3_b = src_b + 3 * src_ldb, *dst_c3_b = dst_b + 3 * dst_ldb;

    for(int p = 0; p < k; ++p){
        *dst_c0_b++ = *src_c0_b++;
        *dst_c1_b++ = *src_c1_b++;
        *dst_c2_b++ = *src_c2_b++;
        *dst_c3_b++ = *src_c3_b++;
    }
}

void kernel128X128(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta, bool do_cache_a, bool do_cache_b, bool do_cache_b_now){

    float* real_a = nullptr, *real_b = nullptr;
    
    if(do_cache_a){
        real_a = static_cast<float*>(malloc(m * k * sizeof(float)));
        if(nullptr == a){
            cout << "malloc failed! Line: "<< __LINE__ << endl; exit(1);
        }
    }
    else{
        real_a = a;
    }

    if(do_cache_b){
        static float* b_cache = static_cast<float*>(malloc(k * n * sizeof(float)));
        if(nullptr == b_cache){
            cout << "malloc failed! Line: "<< __LINE__ << endl; exit(1);
        }
        real_b = b_cache;
    }
    else{
        real_b = b;
    }


    for(int col = 0; col < n; col += 4){
        if(do_cache_b && do_cache_b_now){
            pack_b(k, &b[cord(0, col, ldb)], ldb, &real_b[col * k], k);
        }

        for(int row = 0; row < m; row += 4){
            if(do_cache_a && 0 == col){
                pack_a(k, &a[cord(row, 0, lda)], lda, &real_a[row], m);
            }

            float* a_head = (do_cache_a) ? &real_a[row] : &real_a[cord(row, 0, lda)];
            float* b_head = (do_cache_b) ? &real_b[k * col] : &real_b[cord(0, col, ldb)];
            float* c_head = &c[cord(row, col, ldc)];
            int real_lda = (do_cache_a) ? m : lda;
            int real_ldb = (do_cache_b) ? k : ldb;
            
            addDot4X4VReg(k, a_head, real_lda, b_head, real_ldb, c_head, ldc, alpha, beta);
        }
    }

    if(do_cache_a){
        free(real_a); real_a = nullptr;
    }
 
}

void gemm_cpu_per_thread(const thread_param_t& param){

    int m = param.m, n = param.n, k = param.k;
    float* a = param.a; int lda = param.lda;
    float* b = param.b; int ldb = param.ldb;
    float* c = param.c; int ldc = param.ldc;
    float alpha = param.alpha, beta = param.beta;

    bool do_cache_a = false, do_cache_b = false, do_cache_b_now = false;
    
    if(m >  m_block) do_cache_a = true;
    if(n > n_block) do_cache_b = true;

    for(int col = 0; col < n; col += n_block){
        for(int row = 0; row < m; row += m_block){
            
            float* a_head = &(a[cord(row, 0, lda)]);
            float* b_head = &(b[cord(0, col, ldb)]);
            float* c_head = &(c[cord(row, col, ldc)]);

            if(0 == row) do_cache_b_now = true;
            else do_cache_b_now = false;
            
            int m_real_block = min(m - row, m_block);
            int n_real_block = min(n - col, n_block);

            do_cache_a = false;
            do_cache_b = false;
            
            kernel128X128(m_real_block, n_real_block, k, a_head, lda, b_head, ldb, c_head, ldc, alpha, beta, do_cache_a, do_cache_b, do_cache_b_now);
        }
    }
    
}




void gemm_cpu(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc, float alpha, float beta){

    if(m % m_block != 0 || n % n_block != 0 || 0 == m || 0 == n){
        cout << "Optimize8: Param m = " << m << " n = " << n << " not implemented!" << endl;
        return;
    }
    
    //int max_phy_thread_num = thread::hardware_concurrency();
    bool is_row_split = true; if(n > m) is_row_split = false;
    int m_block4 = m / m_block, n_block4 = n / n_block;
    int real_thread_num = 6;

    //cout << "max_phy_thread_num = " << max_phy_thread_num << " real_thread_num = " << real_thread_num << endl;


    thread_param_t* params = new thread_param_t[real_thread_num]; 
    
    int offset = (is_row_split) ? (m_block4 / real_thread_num * m_block) : (n_block4 / real_thread_num * n_block);
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

    #pragma omp parallel num_threads(real_thread_num)
    {
        int tid = omp_get_thread_num();
        gemm_cpu_per_thread(params[tid]);
    }
    delete [] params;
    
}


#endif