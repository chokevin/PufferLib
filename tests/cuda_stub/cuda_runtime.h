// Host emulation of the small CUDA surface that src/categorical.cuh uses, so
// the production primitive can be executed and checked on machines without a
// GPU (see tests/categorical_host_exec.cc). A 32-lane warp is emulated with 32
// std::threads and a reusable barrier, so every shuffle / ballot / prefix scan
// runs the real lockstep protocol rather than a rewritten host version.
//
// This is test-only scaffolding: it never ships in a CUDA build, where the real
// <cuda_runtime.h> wins on the include path.
#ifndef PUF_TEST_CUDA_RUNTIME_STUB_H
#define PUF_TEST_CUDA_RUNTIME_STUB_H

#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#define __device__
#define __host__
#define __global__
#define __forceinline__ inline

typedef int cudaError_t;
typedef void* cudaStream_t;
enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 0,
    cudaMemcpyDeviceToHost = 1,
    cudaMemcpyDeviceToDevice = 2,
};
enum { cudaSuccess = 0, cudaHostAllocPortable = 1 };

inline cudaError_t cudaMalloc(void** p, size_t bytes) {
    *p = std::malloc(bytes);
    return 0;
}
inline cudaError_t cudaHostAlloc(void** p, size_t bytes, unsigned) {
    *p = std::malloc(bytes);
    return 0;
}
inline cudaError_t cudaMemset(void* p, int v, size_t bytes) {
    std::memset(p, v, bytes);
    return 0;
}
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t bytes, cudaStream_t) {
    std::memset(p, v, bytes);
    return 0;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes,
        cudaMemcpyKind, cudaStream_t) {
    std::memcpy(dst, src, bytes);
    return 0;
}

// --------------------------------------------------------------------------
// Emulated warp
// --------------------------------------------------------------------------
struct PufHostWarp {
    static const int W = 32;
    std::mutex m;
    std::condition_variable cv;
    int count = 0;
    int generation = 0;
    float fbuf[W];
    int ibuf[W];
    unsigned bits = 0;

    void barrier() {
        std::unique_lock<std::mutex> lock(m);
        int gen = generation;
        if (++count == W) {
            count = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [&] { return generation != gen; });
        }
    }
};

struct PufHostDim3 {
    unsigned x, y, z;
};

extern thread_local PufHostWarp* puf_host_warp;
extern thread_local int puf_host_lane;
extern thread_local PufHostDim3 threadIdx;

inline float __shfl_xor_sync(unsigned, float v, int off, int = 32) {
    PufHostWarp* w = puf_host_warp;
    w->fbuf[puf_host_lane] = v;
    w->barrier();
    float r = w->fbuf[puf_host_lane ^ off];
    w->barrier();
    return r;
}

inline int __shfl_xor_sync(unsigned, int v, int off, int = 32) {
    PufHostWarp* w = puf_host_warp;
    w->ibuf[puf_host_lane] = v;
    w->barrier();
    int r = w->ibuf[puf_host_lane ^ off];
    w->barrier();
    return r;
}

inline float __shfl_up_sync(unsigned, float v, int delta, int = 32) {
    PufHostWarp* w = puf_host_warp;
    w->fbuf[puf_host_lane] = v;
    w->barrier();
    int src = puf_host_lane - delta;
    float r = src >= 0 ? w->fbuf[src] : v;
    w->barrier();
    return r;
}

inline float __shfl_sync(unsigned, float v, int src, int = 32) {
    PufHostWarp* w = puf_host_warp;
    w->fbuf[puf_host_lane] = v;
    w->barrier();
    float r = w->fbuf[src & 31];
    w->barrier();
    return r;
}

inline unsigned __ballot_sync(unsigned, bool pred) {
    PufHostWarp* w = puf_host_warp;
    w->ibuf[puf_host_lane] = pred ? 1 : 0;
    w->barrier();
    if (puf_host_lane == 0) {
        unsigned bits = 0;
        for (int i = 0; i < PufHostWarp::W; i++) {
            bits |= (unsigned)(w->ibuf[i] & 1) << i;
        }
        w->bits = bits;
    }
    w->barrier();
    unsigned out = w->bits;
    w->barrier();
    return out;
}

inline void __syncwarp(unsigned = 0xffffffffu) {
    puf_host_warp->barrier();
}

inline int __ffs(int v) {
    return v == 0 ? 0 : __builtin_ffs(v);
}

inline float __expf(float x) { return expf(x); }
inline float __logf(float x) { return logf(x); }

int puf_host_atomic_cas(int* addr, int cmp, int val);
inline int atomicCAS(int* addr, int cmp, int val) {
    return puf_host_atomic_cas(addr, cmp, val);
}

// Run `body(lane)` on all 32 emulated lanes of one warp.
void puf_host_warp_run(const std::function<void(int)>& body);

#endif  // PUF_TEST_CUDA_RUNTIME_STUB_H
