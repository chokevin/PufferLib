// Host stub for <cuda_bf16.h> used by tests/categorical_host_exec.cc.
// Implements bfloat16 storage with the same round-to-nearest-even conversion
// nvcc's __float2bfloat16 performs, so the host test exercises the same
// precision behaviour as a BF16 build.
#ifndef PUF_TEST_CUDA_BF16_STUB_H
#define PUF_TEST_CUDA_BF16_STUB_H

#include <cstdint>
#include <cstring>

struct __nv_bfloat16 {
    uint16_t raw;
};

inline float __bfloat162float(__nv_bfloat16 v) {
    uint32_t bits = (uint32_t)v.raw << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline __nv_bfloat16 __float2bfloat16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    __nv_bfloat16 out;
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        out.raw = (uint16_t)((bits >> 16) | 0x0040u);  // quiet NaN
        return out;
    }
    uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    out.raw = (uint16_t)(rounded >> 16);
    return out;
}

#endif  // PUF_TEST_CUDA_BF16_STUB_H
