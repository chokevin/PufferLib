// Minimal CUDA-ish stubs so the extracted kernels can be type-checked by a
// host C++ compiler. Not for execution.
#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static
#define __syncthreads() ((void)0)

struct Dim3 { int x, y, z; };
static Dim3 blockIdx, threadIdx, blockDim, gridDim;

typedef float precision_t;
static inline float to_float(float x) { return x; }
static inline float from_float(float x) { return x; }
static inline float __expf(float x) { return expf(x); }
static inline float __logf(float x) { return logf(x); }
static inline int atomicCAS(int* p, int cmp, int val) {
    if (*p == cmp) { *p = val; return cmp; }
    return *p;
}

#define PUF_MAX_DIMS 8
typedef struct { float* data; long long shape[PUF_MAX_DIMS]; } Float;
typedef struct { int* data; long long shape[PUF_MAX_DIMS]; } Int;
typedef struct { precision_t* data; long long shape[PUF_MAX_DIMS]; } Prec;

static inline long numel(long long* shape) {
    long n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i]; i++) n *= (long)shape[i];
    return n;
}

// Allocator stub (register_ppo_buffers).
typedef struct { int unused; } Allocator;
static inline void alloc_register(Allocator*, Float*) {}
static inline void alloc_register(Allocator*, Int*) {}
static inline int cudaMalloc(void** p, size_t n) { *p = malloc(n); return 0; }
static inline int cudaMemset(void* p, int v, size_t n) { memset(p, v, n); return 0; }

// Block reduce from pufferl.cu (verbatim contract).
static inline void block_reduce_sum(float* smem, float* out, int tid,
        int nthreads, int nchan) {
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < nchan; c++) {
                smem[c * nthreads + tid] += smem[c * nthreads + tid + s];
            }
        }
    }
    if (tid == 0) {
        for (int c = 0; c < nchan; c++) out[c] = smem[c * nthreads];
    }
}

static inline float finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) return 0.0f;
    if (isinf(x)) return x > 0.0f ? hi : lo;
    return fminf(hi, fmaxf(lo, x));
}
static inline float safe_continuous_mean(const precision_t* l, int i) {
    return finite_or_clamp(to_float(l[i]), -1.0e6f, 1.0e6f);
}
static inline float safe_continuous_logstd(const precision_t* l, int i) {
    return finite_or_clamp(to_float(l[i]), -20.0f, 2.0f);
}

// Env constants: a 4-head categorical build.
#define NUM_ATNS 4
#define ACT_SIZES {3, 2, 4, 2}
constexpr int PPO_THREADS = 256;

// curand stubs for the sampler.
struct curandStatePhilox4_32_10_t { unsigned long long s; };
static inline float curand_uniform(curandStatePhilox4_32_10_t*) { return 0.5f; }
static inline float curand_normal(curandStatePhilox4_32_10_t*) { return 0.0f; }
