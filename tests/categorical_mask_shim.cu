// Adversarial test shim around the production categorical primitive
// (src/categorical.cuh). Mirrors how src/pufferl.cu (rollout sampling) and
// src/algo.cu (learner cache + gradients) drive the primitive, without any
// environment header, so the numeric and error semantics can be checked in
// isolation. Built by tests/test_categorical_mask.py with nvcc, once as float
// (-DPRECISION_FLOAT) and once as bfloat16.
//
//   nvcc -shared -Xcompiler -fPIC -O2 -std=c++17 --fmad=false \
//        -Xcompiler -ffp-contract=off -o categorical_shim.so \
//        tests/categorical_mask_shim.cu
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <string.h>

#include "../src/categorical.cuh"

#define SHIM_GUARD 64
#define SHIM_CANARY 0x7fc0dead

static __global__ void shim_cast_logits(precision_t* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = from_float(src[i]);
    }
}

static __global__ void shim_cast_mask(precision_t* dst, const unsigned char* src,
        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = from_float((float)src[i]);
    }
}

// Mirrors cache_imp_and_v's discrete body (no env head gating).
static __global__ void shim_cache_kernel(
        const precision_t* __restrict__ logits,
        const precision_t* __restrict__ mask,
        const int* __restrict__ act_sizes,
        const float* __restrict__ actions,
        const float* __restrict__ old_logprobs,
        float* __restrict__ logps, float* __restrict__ new_lp_out,
        float* __restrict__ entropy_out, float* __restrict__ imp_out,
        int num_heads, int A_total, int rows, PufCatStatus* status) {
    const int lane = puf_cat_lane();
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / PUF_CAT_WARP;
    if (row >= rows) {
        return;
    }
    int at_base = row * A_total;
    float old_lp = old_logprobs[row];
    int err_code = isfinite(old_lp) ? PUF_CAT_OK : PUF_CAT_ERR_OLD_LOGPROB;
    int err_head = -1;
    int err_detail = -1;
    float new_lp = 0.0f;
    float entropy = 0.0f;
    int offset = 0;
    for (int h = 0; h < num_heads && err_code == PUF_CAT_OK; h++) {
        int A = act_sizes[h];
        const precision_t* head_logits = logits + at_base + offset;
        const precision_t* head_mask = mask + at_base + offset;
        float* head_logps = logps + at_base + offset;
        PufCatNorm norm = puf_cat_head_norm(head_logits, head_mask, A);
        if (norm.code != PUF_CAT_OK) {
            err_code = norm.code;
            err_head = h;
            err_detail = norm.detail;
            break;
        }
        int act_code = PUF_CAT_OK;
        int act = puf_cat_check_action(actions[row * num_heads + h], A, &act_code);
        if (act_code != PUF_CAT_OK) {
            err_code = act_code;
            err_head = h;
            break;
        }
        if (!puf_cat_legal(head_mask[act])) {
            err_code = PUF_CAT_ERR_ACTION_ILLEGAL;
            err_head = h;
            err_detail = act;
            break;
        }
        puf_cat_write_logps(head_logits, head_mask, A, norm, head_logps);
        new_lp += puf_cat_logp_at(head_logits, act, norm);
        entropy += puf_cat_entropy_from_logps(head_logps, head_mask, A);
        offset += A;
    }
    if (err_code != PUF_CAT_OK) {
        for (int a = lane; a < A_total; a += PUF_CAT_WARP) {
            logps[at_base + a] = 0.0f;
        }
        if (lane == 0) {
            new_lp_out[row] = isfinite(old_lp) ? old_lp : 0.0f;
            entropy_out[row] = 0.0f;
            imp_out[row] = 1.0f;
            puf_cat_report(status, err_code, row, err_head, err_detail);
        }
        return;
    }
    if (lane == 0) {
        new_lp_out[row] = new_lp;
        entropy_out[row] = entropy;
        imp_out[row] = __expf(new_lp - old_lp);
    }
}

// Mirrors sample_logits' discrete body with a caller-supplied uniform stream.
static __global__ void shim_sample_kernel(
        const precision_t* __restrict__ logits,
        const precision_t* __restrict__ mask,
        const int* __restrict__ act_sizes,
        const float* __restrict__ uniforms, float eps,
        float* __restrict__ actions_out, float* __restrict__ logp_out,
        int num_heads, int A_total, int rows, PufCatStatus* status) {
    const int lane = puf_cat_lane();
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / PUF_CAT_WARP;
    if (row >= rows) {
        return;
    }
    int at_base = row * A_total;
    int err_code = PUF_CAT_OK;
    int err_head = -1;
    int err_detail = -1;
    float total_logp = 0.0f;
    int offset = 0;
    for (int h = 0; h < num_heads; h++) {
        int A = act_sizes[h];
        const precision_t* head_logits = logits + at_base + offset;
        const precision_t* head_mask = mask + at_base + offset;
        PufCatNorm norm = puf_cat_head_norm(
            head_logits, head_mask, A, eps > 0.0f);
        if (norm.code != PUF_CAT_OK) {
            err_code = norm.code;
            err_head = h;
            err_detail = norm.detail;
            break;
        }
        float inv_legal = eps > 0.0f ? 1.0f / (float)norm.legal_count : 0.0f;
        float u = uniforms[row * num_heads + h];
        int sampled = puf_cat_sample(head_logits, head_mask, A, norm, u,
            eps, inv_legal);
        float logp = puf_cat_logp_at(head_logits, sampled, norm);
        if (eps > 0.0f) {
            logp = logf((1.0f - eps) * expf(logp) + eps * inv_legal);
        }
        if (lane == 0) {
            actions_out[row * num_heads + h] = (float)sampled;
        }
        total_logp += logp;
        offset += A;
    }
    if (err_code != PUF_CAT_OK) {
        for (int h = lane; h < num_heads; h += PUF_CAT_WARP) {
            actions_out[row * num_heads + h] = 0.0f;
        }
        if (lane == 0) {
            logp_out[row] = 0.0f;
            puf_cat_report(status, err_code, row, err_head, err_detail);
        }
        return;
    }
    if (lane == 0) {
        logp_out[row] = total_logp;
    }
}

// Mirrors ppo_categorical_grad (no env head gating).
static __global__ void shim_grad_kernel(
        float* __restrict__ grad, const precision_t* __restrict__ mask,
        const int* __restrict__ act_sizes, const float* __restrict__ actions,
        const float* __restrict__ dlogp, float d_entropy_term,
        int num_heads, int A_total, int rows) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / PUF_CAT_WARP;
    if (row >= rows) {
        return;
    }
    int at_base = row * A_total;
    float d_new_logp = dlogp[row];
    int offset = 0;
    for (int h = 0; h < num_heads; h++) {
        int A = act_sizes[h];
        float* head_grad = grad + at_base + offset;
        const precision_t* head_mask = mask + at_base + offset;
        int act_code = PUF_CAT_OK;
        int act = puf_cat_check_action(actions[row * num_heads + h], A, &act_code);
        if (act_code != PUF_CAT_OK) {
            puf_cat_zero_head(head_grad, A);
            offset += A;
            continue;
        }
        float entropy = puf_cat_entropy_from_logps(head_grad, head_mask, A);
        __syncwarp();
        puf_cat_write_grads(head_grad, head_mask, A, act, d_new_logp,
            d_entropy_term, entropy);
        offset += A;
    }
}

// --------------------------------------------------------------------------
// Host entry points. Everything is allocated with guard bands so out-of-range
// warp writes are caught.
// --------------------------------------------------------------------------
struct ShimBuf {
    void* base;
    void* data;
    size_t bytes;
};

static ShimBuf shim_alloc(size_t bytes) {
    ShimBuf b;
    b.bytes = bytes;
    cudaMalloc(&b.base, bytes + 2 * SHIM_GUARD);
    cudaMemset(b.base, 0, bytes + 2 * SHIM_GUARD);
    int canary = SHIM_CANARY;
    for (int i = 0; i < SHIM_GUARD / 4; i++) {
        cudaMemcpy((char*)b.base + i * 4, &canary, 4, cudaMemcpyHostToDevice);
        cudaMemcpy((char*)b.base + SHIM_GUARD + bytes + i * 4, &canary, 4,
            cudaMemcpyHostToDevice);
    }
    b.data = (char*)b.base + SHIM_GUARD;
    return b;
}

static int shim_guard_ok(const ShimBuf& b) {
    int host[2 * SHIM_GUARD / 4];
    cudaMemcpy(host, b.base, SHIM_GUARD, cudaMemcpyDeviceToHost);
    cudaMemcpy(host + SHIM_GUARD / 4, (char*)b.base + SHIM_GUARD + b.bytes,
        SHIM_GUARD, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 2 * SHIM_GUARD / 4; i++) {
        if (host[i] != SHIM_CANARY) {
            return 0;
        }
    }
    return 1;
}

static void shim_free(ShimBuf& b) {
    cudaFree(b.base);
    b.base = NULL;
    b.data = NULL;
}

extern "C" {

int puf_cat_shim_uses_bf16() {
    return sizeof(precision_t) == sizeof(float) ? 0 : 1;
}

int puf_cat_shim_device_ok() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 ? 1 : 0;
}

const char* puf_cat_shim_code_name(int code) {
    return puf_cat_code_name(code);
}

// Byte masks are cast to precision_t first, exactly as the trainer does.
int puf_cat_shim_cache(int rows, int num_heads, const int* act_sizes,
        const float* logits, const unsigned char* mask, const float* actions,
        const float* old_logprobs,
        float* logps_out, float* new_lp_out, float* entropy_out,
        float* imp_out, int* status_out) {
    int A_total = 0;
    for (int h = 0; h < num_heads; h++) {
        A_total += act_sizes[h] > 0 ? act_sizes[h] : 0;
    }
    if (A_total <= 0) {
        A_total = 1;
    }
    size_t n = (size_t)rows * A_total;
    ShimBuf d_logits_f = shim_alloc(n * sizeof(float));
    ShimBuf d_logits = shim_alloc(n * sizeof(precision_t));
    ShimBuf d_mask_b = shim_alloc(n);
    ShimBuf d_mask_p = shim_alloc(n * sizeof(precision_t));
    ShimBuf d_sizes = shim_alloc((size_t)num_heads * sizeof(int));
    ShimBuf d_actions = shim_alloc((size_t)rows * num_heads * sizeof(float));
    ShimBuf d_old = shim_alloc((size_t)rows * sizeof(float));
    ShimBuf d_logps = shim_alloc(n * sizeof(float));
    ShimBuf d_new_lp = shim_alloc((size_t)rows * sizeof(float));
    ShimBuf d_entropy = shim_alloc((size_t)rows * sizeof(float));
    ShimBuf d_imp = shim_alloc((size_t)rows * sizeof(float));
    ShimBuf d_status = shim_alloc(sizeof(PufCatStatus));

    cudaMemcpy(d_logits_f.data, logits, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask_b.data, mask, n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes.data, act_sizes, num_heads * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_actions.data, actions, (size_t)rows * num_heads * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_old.data, old_logprobs, (size_t)rows * sizeof(float),
        cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    shim_cast_logits<<<blocks, threads>>>((precision_t*)d_logits.data,
        (const float*)d_logits_f.data, (int)n);
    shim_cast_mask<<<blocks, threads>>>((precision_t*)d_mask_p.data,
        (const unsigned char*)d_mask_b.data, (int)n);

    int warp_blocks = (int)(((size_t)rows * PUF_CAT_WARP + threads - 1) / threads);
    shim_cache_kernel<<<warp_blocks, threads>>>(
        (const precision_t*)d_logits.data,
        (const precision_t*)d_mask_p.data, (const int*)d_sizes.data,
        (const float*)d_actions.data, (const float*)d_old.data,
        (float*)d_logps.data, (float*)d_new_lp.data,
        (float*)d_entropy.data, (float*)d_imp.data,
        num_heads, A_total, rows, (PufCatStatus*)d_status.data);
    int rc = cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;

    cudaMemcpy(logps_out, d_logps.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_lp_out, d_new_lp.data, (size_t)rows * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(entropy_out, d_entropy.data, (size_t)rows * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(imp_out, d_imp.data, (size_t)rows * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(status_out, d_status.data, sizeof(PufCatStatus),
        cudaMemcpyDeviceToHost);

    if (!shim_guard_ok(d_logps) || !shim_guard_ok(d_new_lp)
            || !shim_guard_ok(d_entropy) || !shim_guard_ok(d_imp)) {
        rc = -2;
    }
    shim_free(d_logits_f); shim_free(d_logits); shim_free(d_mask_b);
    shim_free(d_mask_p); shim_free(d_sizes); shim_free(d_actions);
    shim_free(d_old); shim_free(d_logps); shim_free(d_new_lp);
    shim_free(d_entropy); shim_free(d_imp); shim_free(d_status);
    return rc;
}

int puf_cat_shim_sample(int rows, int num_heads, const int* act_sizes,
        const float* logits, const unsigned char* mask, const float* uniforms,
        float eps, float* actions_out, float* logp_out, int* status_out) {
    int A_total = 0;
    for (int h = 0; h < num_heads; h++) {
        A_total += act_sizes[h] > 0 ? act_sizes[h] : 0;
    }
    if (A_total <= 0) {
        A_total = 1;
    }
    size_t n = (size_t)rows * A_total;
    ShimBuf d_logits_f = shim_alloc(n * sizeof(float));
    ShimBuf d_logits = shim_alloc(n * sizeof(precision_t));
    ShimBuf d_mask_b = shim_alloc(n);
    ShimBuf d_mask_p = shim_alloc(n * sizeof(precision_t));
    ShimBuf d_sizes = shim_alloc((size_t)num_heads * sizeof(int));
    ShimBuf d_u = shim_alloc((size_t)rows * num_heads * sizeof(float));
    ShimBuf d_actions = shim_alloc((size_t)rows * num_heads * sizeof(float));
    ShimBuf d_logp = shim_alloc((size_t)rows * sizeof(float));
    ShimBuf d_status = shim_alloc(sizeof(PufCatStatus));

    cudaMemcpy(d_logits_f.data, logits, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask_b.data, mask, n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes.data, act_sizes, num_heads * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.data, uniforms, (size_t)rows * num_heads * sizeof(float),
        cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    shim_cast_logits<<<blocks, threads>>>((precision_t*)d_logits.data,
        (const float*)d_logits_f.data, (int)n);
    shim_cast_mask<<<blocks, threads>>>((precision_t*)d_mask_p.data,
        (const unsigned char*)d_mask_b.data, (int)n);

    int warp_blocks = (int)(((size_t)rows * PUF_CAT_WARP + threads - 1) / threads);
    shim_sample_kernel<<<warp_blocks, threads>>>(
        (const precision_t*)d_logits.data, (const precision_t*)d_mask_p.data,
        (const int*)d_sizes.data, (const float*)d_u.data, eps,
        (float*)d_actions.data, (float*)d_logp.data,
        num_heads, A_total, rows, (PufCatStatus*)d_status.data);
    int rc = cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;

    cudaMemcpy(actions_out, d_actions.data,
        (size_t)rows * num_heads * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(logp_out, d_logp.data, (size_t)rows * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(status_out, d_status.data, sizeof(PufCatStatus),
        cudaMemcpyDeviceToHost);
    if (!shim_guard_ok(d_actions) || !shim_guard_ok(d_logp)) {
        rc = -2;
    }
    shim_free(d_logits_f); shim_free(d_logits); shim_free(d_mask_b);
    shim_free(d_mask_p); shim_free(d_sizes); shim_free(d_u);
    shim_free(d_actions); shim_free(d_logp); shim_free(d_status);
    return rc;
}

int puf_cat_shim_grad(int rows, int num_heads, const int* act_sizes,
        const float* logps, const unsigned char* mask, const float* actions,
        const float* dlogp, float d_entropy_term, float* grad_out) {
    int A_total = 0;
    for (int h = 0; h < num_heads; h++) {
        A_total += act_sizes[h] > 0 ? act_sizes[h] : 0;
    }
    if (A_total <= 0) {
        A_total = 1;
    }
    size_t n = (size_t)rows * A_total;
    ShimBuf d_grad = shim_alloc(n * sizeof(float));
    ShimBuf d_mask_b = shim_alloc(n);
    ShimBuf d_mask_p = shim_alloc(n * sizeof(precision_t));
    ShimBuf d_sizes = shim_alloc((size_t)num_heads * sizeof(int));
    ShimBuf d_actions = shim_alloc((size_t)rows * num_heads * sizeof(float));
    ShimBuf d_dlogp = shim_alloc((size_t)rows * sizeof(float));

    cudaMemcpy(d_grad.data, logps, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask_b.data, mask, n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes.data, act_sizes, num_heads * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_actions.data, actions, (size_t)rows * num_heads * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_dlogp.data, dlogp, (size_t)rows * sizeof(float),
        cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    shim_cast_mask<<<blocks, threads>>>((precision_t*)d_mask_p.data,
        (const unsigned char*)d_mask_b.data, (int)n);

    int warp_blocks = (int)(((size_t)rows * PUF_CAT_WARP + threads - 1) / threads);
    shim_grad_kernel<<<warp_blocks, threads>>>(
        (float*)d_grad.data, (const precision_t*)d_mask_p.data,
        (const int*)d_sizes.data, (const float*)d_actions.data,
        (const float*)d_dlogp.data, d_entropy_term, num_heads, A_total, rows);
    int rc = cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
    cudaMemcpy(grad_out, d_grad.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (!shim_guard_ok(d_grad)) {
        rc = -2;
    }
    shim_free(d_grad); shim_free(d_mask_b); shim_free(d_mask_p);
    shim_free(d_sizes); shim_free(d_actions); shim_free(d_dlogp);
    return rc;
}

}  // extern "C"
