#include <iostream>
#include <random>
#include <immintrin.h>
#include <emmintrin.h>
#include <chrono>
#include <fstream>
#include <iomanip>

#include <intrin.h>
#define NOMINMAX
#include <Windows.h>

size_t N_min = 8, N_max = 4096;
#define DEF_BLOCK_SIZE 64

template <typename T_MAT_DATA>
void resize_mat(T_MAT_DATA*& mat, size_t size) {
    mat = (T_MAT_DATA*)_aligned_malloc(size * size * sizeof(T_MAT_DATA), 64);;
}

template <typename T_MAT_DATA>
void fill_random(T_MAT_DATA* mat, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T_MAT_DATA> dist(30.0, 80.0f);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            mat[i*size + j] = dist(gen);
        }
    }
}

template <typename T_MAT_DATA>
void delete_mat(T_MAT_DATA*& mat, size_t size) {
    _aligned_free(mat);
}

template <typename T_MAT_DATA>
void init_mult(T_MAT_DATA* A, T_MAT_DATA* B, T_MAT_DATA* C, size_t size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                C[i*size + j] += A[i*size + k] * B[k*size + j];
            }
        }
    }
}

template <typename T_MAT_DATA>
int check_mat(T_MAT_DATA* res_1, T_MAT_DATA* res_2, size_t size, const char* string) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            float diff = std::abs(res_1[i*size + j] - res_2[i*size + j]);
            float mag = std::max(std::abs(res_1[i*size + j]), std::abs(res_2[i*size + j]));
            if (diff > mag * 1e-2f + 1e-3f) {
                std::cout << string << " mismatch at [" << i << "][" << j
                    << "]: " << res_1[i*size + j] << " vs " << res_2[i*size + j]
                    << " (diff=" << diff << ")" << std::endl;
                return 1;
            }
        }
    }
    return 0;
}

template <typename T_MAT_DATA>
void mat_to_null(T_MAT_DATA* mat, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            mat[i* size + j] = 0.0f;
        }
    }
}

#define SSE_BLOCKNOHOR_STEP(off, acc_lo, acc_hi) { \
    __m128 a_##acc_lo = _mm_broadcast_ss(&Az[x*DEF_BLOCK_SIZE + k + off]); \
    __m256 bc_##acc_lo = _mm256_load_ps(&Bz[(k + off)*DEF_BLOCK_SIZE + y]); \
    acc_lo = _mm_fmadd_ps(a_##acc_lo, _mm256_castps256_ps128(bc_##acc_lo), acc_lo); \
    acc_hi = _mm_fmadd_ps(a_##acc_lo, _mm256_extractf128_ps(bc_##acc_lo, 1), acc_hi); \
}

void sse_block_no_horizontal_unroll4(float* A, float* B, float* C, size_t N) {
    const int BLOCK_SIZE = std::min(DEF_BLOCK_SIZE, int(N));
    __declspec(align(32)) float Az[DEF_BLOCK_SIZE * DEF_BLOCK_SIZE];
    __declspec(align(32)) float Bz[DEF_BLOCK_SIZE * DEF_BLOCK_SIZE];
    __m256 zero = _mm256_setzero_ps();
    for (size_t x0 = 0; x0 < N; x0 += BLOCK_SIZE) {
        for (size_t y0 = 0; y0 < N; y0 += BLOCK_SIZE) {
            __declspec(align(32)) float sum[DEF_BLOCK_SIZE * DEF_BLOCK_SIZE];
            for (size_t x = 0; x < BLOCK_SIZE; x++)
                for (size_t y = 0; y < BLOCK_SIZE; y += 8)
                    _mm256_store_ps(&sum[x * DEF_BLOCK_SIZE + y], zero);
            size_t max_z = N / BLOCK_SIZE;
            for (size_t z = 0; z < max_z; z++) {
                size_t zb = z * BLOCK_SIZE;
                for (size_t x = 0; x < BLOCK_SIZE; x++)
                    for (size_t y = 0; y < BLOCK_SIZE; y++) {
                        Az[x * DEF_BLOCK_SIZE + y] = A[(x0 + x) * N + zb + y];
                        Bz[x * DEF_BLOCK_SIZE + y] = B[(zb + x) * N + y0 + y];
                    }
                for (size_t x = 0; x < BLOCK_SIZE; x++) {
                    for (size_t y = 0; y < BLOCK_SIZE; y += 8) {
                        __m128 s0_lo = _mm_setzero_ps(), s0_hi = _mm_setzero_ps();
                        __m128 s1_lo = _mm_setzero_ps(), s1_hi = _mm_setzero_ps();
                        __m128 s2_lo = _mm_setzero_ps(), s2_hi = _mm_setzero_ps();
                        __m128 s3_lo = _mm_setzero_ps(), s3_hi = _mm_setzero_ps();
                        for (size_t k = 0; k < BLOCK_SIZE; k += 4) {
                            SSE_BLOCKNOHOR_STEP(0, s0_lo, s0_hi);
                            SSE_BLOCKNOHOR_STEP(1, s1_lo, s1_hi);
                            SSE_BLOCKNOHOR_STEP(2, s2_lo, s2_hi);
                            SSE_BLOCKNOHOR_STEP(3, s3_lo, s3_hi);
                        }
                        __m128 s_lo = _mm_add_ps(_mm_add_ps(s0_lo, s1_lo), _mm_add_ps(s2_lo, s3_lo));
                        __m128 s_hi = _mm_add_ps(_mm_add_ps(s0_hi, s1_hi), _mm_add_ps(s2_hi, s3_hi));
                        __m256 s = _mm256_set_m128(s_hi, s_lo);
                        __m256 tmp_s = _mm256_load_ps(&sum[x * DEF_BLOCK_SIZE + y]);
                        _mm256_store_ps(&sum[x * DEF_BLOCK_SIZE + y], _mm256_add_ps(tmp_s, s));
                    }
                }
            }
            for (size_t x = 0; x < BLOCK_SIZE; x++)
                for (size_t y = 0; y < BLOCK_SIZE; y += 8)
                    _mm256_store_ps(&C[(x0 + x) * N + y0 + y], _mm256_load_ps(&sum[x * DEF_BLOCK_SIZE + y]));
        }
    }
}

#define SSE_BUFNOHOR_STEP(off, acc_lo, acc_hi) { \
    __m128 a_##acc_lo = _mm_broadcast_ss(&A[i*N + k + off]); \
    __m256 bc_##acc_lo = _mm256_load_ps(&AvxBuf[(k + off) * 8]); \
    acc_lo = _mm_fmadd_ps(a_##acc_lo, _mm256_castps256_ps128(bc_##acc_lo), acc_lo); \
    acc_hi = _mm_fmadd_ps(a_##acc_lo, _mm256_extractf128_ps(bc_##acc_lo, 1), acc_hi); \
}

void sse_buffered_no_horizontal_unroll4(float* A, float* B, float* C, size_t N) {
    float* AvxBuf = (float*)_aligned_malloc(8 * N * sizeof(float), 32);

    for (size_t j = 0; j < N; j += 8) {
        for (size_t k = 0; k < N; k++) {
            _mm256_store_ps(&AvxBuf[k * 8], _mm256_load_ps(&B[k * N + j]));
        }
        for (size_t i = 0; i < N; i++) {
            __m128 s0_lo = _mm_setzero_ps(), s0_hi = _mm_setzero_ps();
            __m128 s1_lo = _mm_setzero_ps(), s1_hi = _mm_setzero_ps();
            __m128 s2_lo = _mm_setzero_ps(), s2_hi = _mm_setzero_ps();
            __m128 s3_lo = _mm_setzero_ps(), s3_hi = _mm_setzero_ps();

            for (size_t k = 0; k < N; k += 4) {
                SSE_BUFNOHOR_STEP(0, s0_lo, s0_hi);
                SSE_BUFNOHOR_STEP(1, s1_lo, s1_hi);
                SSE_BUFNOHOR_STEP(2, s2_lo, s2_hi);
                SSE_BUFNOHOR_STEP(3, s3_lo, s3_hi);
            }

            __m128 s_lo = _mm_add_ps(_mm_add_ps(s0_lo, s1_lo), _mm_add_ps(s2_lo, s3_lo));
            __m128 s_hi = _mm_add_ps(_mm_add_ps(s0_hi, s1_hi), _mm_add_ps(s2_hi, s3_hi));
            _mm256_store_ps(&C[i * N + j], _mm256_set_m128(s_hi, s_lo));
        }
    }
    _aligned_free(AvxBuf);
}

#define SSE_NAIVENOHOR_STEP(off, acc_lo, acc_hi) { \
    __m128 a_##acc_lo = _mm_broadcast_ss(&A[i*N + k + off]); \
    __m256 bc_##acc_lo = _mm256_load_ps(&B[(k + off)*N + j]); \
    acc_lo = _mm_fmadd_ps(a_##acc_lo, _mm256_castps256_ps128(bc_##acc_lo), acc_lo); \
    acc_hi = _mm_fmadd_ps(a_##acc_lo, _mm256_extractf128_ps(bc_##acc_lo, 1), acc_hi); \
}

void sse_naive_no_horizontal_unroll4(float* A, float* B, float* C, size_t N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j += 8) {
            __m128 s0_lo = _mm_setzero_ps(), s0_hi = _mm_setzero_ps();
            __m128 s1_lo = _mm_setzero_ps(), s1_hi = _mm_setzero_ps();
            __m128 s2_lo = _mm_setzero_ps(), s2_hi = _mm_setzero_ps();
            __m128 s3_lo = _mm_setzero_ps(), s3_hi = _mm_setzero_ps();
            for (size_t k = 0; k < N; k += 4) {
                SSE_NAIVENOHOR_STEP(0, s0_lo, s0_hi);
                SSE_NAIVENOHOR_STEP(1, s1_lo, s1_hi);
                SSE_NAIVENOHOR_STEP(2, s2_lo, s2_hi);
                SSE_NAIVENOHOR_STEP(3, s3_lo, s3_hi);
            }
            __m128 s_lo = _mm_add_ps(_mm_add_ps(s0_lo, s1_lo), _mm_add_ps(s2_lo, s3_lo));
            __m128 s_hi = _mm_add_ps(_mm_add_ps(s0_hi, s1_hi), _mm_add_ps(s2_hi, s3_hi));
            _mm256_store_ps(&C[i * N + j], _mm256_set_m128(s_hi, s_lo));
        }
    }
}

int check_cpu_specs(std::ofstream& out) {
    int cpuInfo[4] = {};
    try {
        __cpuid(cpuInfo, 0x80000000);
        if (cpuInfo[0] < 0x80000004) throw;

        char brand[50] = {};
        for (int i = 0; i < 3; i++) {
            __cpuid(cpuInfo, 0x80000002 + i);
            memcpy(brand + i * 16, cpuInfo, 16);
        }
        brand[49] = '\0';
        out << "# Cpu_name: " << std::string(brand) << "\n";

        __cpuid(cpuInfo, 1);
        bool cpu_avx = (cpuInfo[2] >> 28) & 1;
        bool os_xsave = (cpuInfo[2] >> 27) & 1;
        bool cpu_fma = (cpuInfo[2] >> 12) & 1;

        if (!cpu_avx || !os_xsave) {
            out << "# AVX is not supported by CPU\n";
            return 2;
        }

        unsigned long long xcr0 = _xgetbv(0);
        if ((xcr0 & 0x6) != 0x6) {
            out << "# AVX is not enabled by OS\n";
            return 2;
        }

        if (!cpu_fma) {
            out << "# FMA is not supported by CPU\n";
            return 3;
        }

        return 0;
    }
    catch (...) {
        out << "# __cpuid didn't work\n";
        return 1;
    }
}

void main_routine(float* A, float* B, float* C_check, float* C) {

    typedef void (*MultFunc)(float*, float*, float*, size_t);
    struct method_struct {
        MultFunc func;
        std::string name;
        size_t min_n;
    };

    method_struct methods[] = {
       // {avx_naive_no_horizontal,           "avx_naive_no_horizontal",           8},
        {sse_naive_no_horizontal_unroll4,   "avx_naive_no_horizontal_unroll4",   8},
       // {avx_naive_no_horizontal_unroll8,   "avx_naive_no_horizontal_unroll8",   8},
       // {avx_buffered_no_horizontal,        "avx_buffered_no_horizontal",        8},
        {sse_buffered_no_horizontal_unroll4,"avx_buffered_no_horizontal_unroll4",8},
       // {avx_buffered_no_horizontal_unroll8,"avx_buffered_no_horizontal_unroll8",8},
       // {avx_block_no_horizontal,           "avx_block_no_horizontal",           8},
        {sse_block_no_horizontal_unroll4,   "avx_block_no_horizontal_unroll4",   8},
       // {avx_block_no_horizontal_unroll8,   "avx_block_no_horizontal_unroll8",   8},
    };

    std::ofstream out("results_avx.txt");

    if (check_cpu_specs(out)) return;
    
    out << "# Block size: " << DEF_BLOCK_SIZE << "\n";
    out << "Size,Method,GFLOPS\n";


    auto bench_start = std::chrono::high_resolution_clock::now();

    for (size_t curr_size = N_min; curr_size <= N_max; curr_size *= 2) {
        std::cout << "Size: " << curr_size << " of " << N_max << "\n";
        resize_mat<float>(A, curr_size);
        resize_mat<float>(B, curr_size);
        resize_mat<float>(C_check, curr_size);
        resize_mat<float>(C, curr_size);

        fill_random<float>(A, curr_size);
        fill_random<float>(B, curr_size);

        mat_to_null(C_check, curr_size);
        init_mult<float>(A, B, C_check, curr_size);

        double flop = 2.0 * curr_size * curr_size * curr_size;

        for (const auto& method : methods) {
            if (curr_size < method.min_n) {
                out << curr_size << "," << method.name << ",SKIP\n";
                continue;
            }
            mat_to_null<float>(C, curr_size);

            int repeats = (curr_size <= 64) ? 1000 : (curr_size <= 256) ? 10 : 1;
            auto start = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < repeats; r++) {
                method.func(A, B, C, curr_size);
            }
            auto end = std::chrono::high_resolution_clock::now();
            double time_sec = std::chrono::duration<double>(end - start).count();
            double gflops = (time_sec > 0.0) ? (flop * repeats / time_sec / 1e9) : 0.0;

            if (check_mat<float>(C_check, C, curr_size, method.name.c_str()) == 0) {
                out << curr_size << "," << method.name << "," << std::fixed << std::setprecision(2) << gflops << "\n";
            }
            else {
                out << curr_size << "," << method.name << ",ERROR\n";
            }
        }

        delete_mat<float>(A, curr_size);
        delete_mat<float>(B, curr_size);
        delete_mat<float>(C_check, curr_size);
        delete_mat<float>(C, curr_size);
    }
    auto bench_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = bench_end - bench_start;
    out << "# Bench time: " << dur.count() << " s";
    out.close();
}

int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);

    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    float* A = nullptr;
    float* B = nullptr;
    float* C_check = nullptr;
    float* C = nullptr;
    main_routine(A, B, C_check, C);
    return 0;
}