// Exact categorical (action-mask) primitive shared by rollout sampling and the
// PPO learner. Environment independent: no NUM_ATNS / ACT_SIZES / env hooks.
// Included by src/pufferl.cu after the precision typedefs and ENV_HEADER and
// before the textual include of src/algo.cu, so both the sampler and the
// learner compile against one normalization implementation.
//
// Semantics (identical in rollout and learner, hence bitwise identical results
// for an unchanged policy):
//   legal(a)  <=> mask[a] != 0            (zero byte illegal, any nonzero legal)
//   max       =  max over legal logits
//   sum       =  sum over legal exp(logit - max)
//   logp(a)   =  (logit[a] - max) - log(sum)          for legal a
//   p(a)      =  exp(logp(a))                          for legal a
//   masked a  =  exactly +0.0f probability, entropy and gradient contribution
// Masked logits are never read into any reduction, so masked NaN / +-Inf /
// FLT_MAX values cannot perturb a legal result and legal logits below -1e4
// cannot lose to a masked value.
//
// Layout: one warp per row, each head scanned in 32-category tiles with warp
// shuffles / ballots / prefix scans. No __syncthreads(), no width-sized or
// head-count-sized thread-local arrays, no "width <= 32" assumption. Every
// lane of the warp must call these helpers (uniform row guard in the caller).
#ifndef PUF_CATEGORICAL_CUH
#define PUF_CATEGORICAL_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Standalone builds (test shims) that include this header without pufferl.cu's
// precision preamble get the same typedefs/macros here.
#ifndef to_float
#ifdef PRECISION_FLOAT
typedef float precision_t;
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif
#endif

#define PUF_CAT_WARP 32
#define PUF_CAT_ALL_LANES 0xffffffffu

// First-error payload. One instance per independent stream of work (rollout
// buffer / learner), reset at a safe launch boundary and read on the host
// before any consumer (env step, GAE, loss, backward, optimizer) can run.
enum PufCatCode {
    PUF_CAT_OK = 0,
    PUF_CAT_ERR_WIDTH = 1,            // head width <= 0
    PUF_CAT_ERR_EMPTY = 2,            // no legal category in the head
    PUF_CAT_ERR_LEGAL_NONFINITE = 3,  // NaN / +-Inf logit on a legal category
    PUF_CAT_ERR_ACTION = 4,           // nonfinite / fractional / out of range
    PUF_CAT_ERR_ACTION_ILLEGAL = 5,   // stored action points at a masked class
    PUF_CAT_ERR_OLD_LOGPROB = 6,      // nonfinite stored PPO old logprob
    PUF_CAT_NUM_CODES = 7,
};

struct PufCatStatus {
    int code;    // PufCatCode; 0 == OK. atomicCAS keeps the first failure.
    int row;     // offending row (agent/timestep)
    int head;    // offending action head, -1 when row scoped
    int detail;  // category index / width / -1
};

__host__ __device__ __forceinline__ const char* puf_cat_code_name(int code) {
    switch (code) {
        case PUF_CAT_OK: return "ok";
        case PUF_CAT_ERR_WIDTH: return "nonpositive action-head width";
        case PUF_CAT_ERR_EMPTY: return "empty legal action set";
        case PUF_CAT_ERR_LEGAL_NONFINITE: return "nonfinite legal logit";
        case PUF_CAT_ERR_ACTION: return "invalid stored action";
        case PUF_CAT_ERR_ACTION_ILLEGAL: return "stored action is masked out";
        case PUF_CAT_ERR_OLD_LOGPROB: return "nonfinite old logprob";
        default: return "unknown categorical error";
    }
}

// First writer wins; payload is only read after a device sync.
__device__ __forceinline__ void puf_cat_report(PufCatStatus* status,
        int code, int row, int head, int detail) {
    if (status == NULL || code == PUF_CAT_OK) {
        return;
    }
    if (atomicCAS(&status->code, PUF_CAT_OK, code) == PUF_CAT_OK) {
        status->row = row;
        status->head = head;
        status->detail = detail;
    }
}

// Zero is illegal, any nonzero value is legal. Environments write byte masks
// (0 / nonzero); the trainer casts them to precision_t, and every nonzero byte
// stays nonzero in float and bfloat16.
__device__ __forceinline__ bool puf_cat_legal(precision_t m) {
    return to_float(m) != 0.0f;
}

// Blocks are warp-aligned (BLOCK_SIZE / PPO_THREADS are multiples of 32).
__device__ __forceinline__ int puf_cat_lane() {
    return threadIdx.x & (PUF_CAT_WARP - 1);
}

__device__ __forceinline__ float puf_cat_warp_max(float v) {
    for (int off = PUF_CAT_WARP / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(PUF_CAT_ALL_LANES, v, off));
    }
    return v;
}

__device__ __forceinline__ float puf_cat_warp_sum(float v) {
    for (int off = PUF_CAT_WARP / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(PUF_CAT_ALL_LANES, v, off);
    }
    return v;
}

__device__ __forceinline__ int puf_cat_warp_sum_int(int v) {
    for (int off = PUF_CAT_WARP / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(PUF_CAT_ALL_LANES, v, off);
    }
    return v;
}

__device__ __forceinline__ int puf_cat_warp_max_int(int v) {
    for (int off = PUF_CAT_WARP / 2; off > 0; off >>= 1) {
        int o = __shfl_xor_sync(PUF_CAT_ALL_LANES, v, off);
        v = o > v ? o : v;
    }
    return v;
}

__device__ __forceinline__ int puf_cat_warp_min_int(int v) {
    for (int off = PUF_CAT_WARP / 2; off > 0; off >>= 1) {
        int o = __shfl_xor_sync(PUF_CAT_ALL_LANES, v, off);
        v = o < v ? o : v;
    }
    return v;
}

// Hillis-Steele inclusive prefix sum across the warp.
__device__ __forceinline__ float puf_cat_warp_prefix_incl(float v) {
    const int lane = puf_cat_lane();
    for (int off = 1; off < PUF_CAT_WARP; off <<= 1) {
        float up = __shfl_up_sync(PUF_CAT_ALL_LANES, v, off);
        if (lane >= off) {
            v += up;
        }
    }
    return v;
}

// Legal-only normalization of one head. Uniform result across the warp.
struct PufCatNorm {
    float max_logit;   // max over legal logits
    float log_sum;     // log(sum(exp(legal logit - max_logit)))
    int legal_count;
    int last_legal;    // highest legal category index
    int code;          // PufCatCode
    int detail;        // offending category / width
};

__device__ PufCatNorm puf_cat_head_norm(
        const precision_t* __restrict__ logits,
        const precision_t* __restrict__ mask, int width,
        bool allow_negative_infinity = false) {
    PufCatNorm n;
    n.max_logit = 0.0f;
    n.log_sum = 0.0f;
    n.legal_count = 0;
    n.last_legal = -1;
    n.code = PUF_CAT_OK;
    n.detail = -1;
    if (width <= 0) {
        n.code = PUF_CAT_ERR_WIDTH;
        n.detail = width;
        return n;
    }

    const int lane = puf_cat_lane();
    float lane_max = -INFINITY;
    int lane_count = 0;
    int lane_finite_count = 0;
    int lane_last = -1;
    int lane_bad = INT_MAX;
    int lane_negative_infinity = INT_MAX;
    for (int base = 0; base < width; base += PUF_CAT_WARP) {
        int a = base + lane;
        if (a < width && puf_cat_legal(mask[a])) {
            float l = to_float(logits[a]);
            lane_count += 1;
            lane_last = a;
            if (isfinite(l)) {
                lane_finite_count += 1;
                lane_max = fmaxf(lane_max, l);
            } else if (allow_negative_infinity && isinf(l) && l < 0.0f) {
                lane_negative_infinity = a < lane_negative_infinity
                    ? a : lane_negative_infinity;
            } else if (a < lane_bad) {
                lane_bad = a;
            }
        }
    }
    n.legal_count = puf_cat_warp_sum_int(lane_count);
    int finite_count = puf_cat_warp_sum_int(lane_finite_count);
    n.last_legal = puf_cat_warp_max_int(lane_last);
    int bad = puf_cat_warp_min_int(lane_bad);
    if (n.legal_count == 0) {
        n.code = PUF_CAT_ERR_EMPTY;
        n.detail = width;
        return n;
    }
    if (bad != INT_MAX) {
        n.code = PUF_CAT_ERR_LEGAL_NONFINITE;
        n.detail = bad;
        return n;
    }
    // Nethack verb-epsilon gives every legal verb positive final mixture mass,
    // so an individual -Inf base logit is a valid zero-softmax-probability
    // input. At least one finite legal logit is still required to define the
    // base component when epsilon is less than one.
    if (finite_count == 0) {
        n.code = PUF_CAT_ERR_LEGAL_NONFINITE;
        n.detail = puf_cat_warp_min_int(lane_negative_infinity);
        return n;
    }

    const float max_logit = puf_cat_warp_max(lane_max);
    float lane_sum = 0.0f;
    for (int base = 0; base < width; base += PUF_CAT_WARP) {
        int a = base + lane;
        float e = 0.0f;
        if (a < width && puf_cat_legal(mask[a])) {
            e = expf(to_float(logits[a]) - max_logit);
        }
        lane_sum += e;
    }
    n.max_logit = max_logit;
    // sum >= 1 (the max category contributes exactly 1), so log_sum is finite.
    n.log_sum = logf(puf_cat_warp_sum(lane_sum));
    return n;
}

// The one expression every path uses for a legal category's log probability.
__device__ __forceinline__ float puf_cat_logp(float logit, const PufCatNorm& n) {
    return (logit - n.max_logit) - n.log_sum;
}

__device__ __forceinline__ float puf_cat_logp_at(
        const precision_t* __restrict__ logits, int a, const PufCatNorm& n) {
    return puf_cat_logp(to_float(logits[a]), n);
}

// Exact operation order for an epsilon mixture with uniform-over-legal mass.
// Rollout old logprob, learner recomputation and policy-gradient scaling all
// call this helper so an unchanged mixed policy produces an exact ratio of one.
struct PufCatUniformMix {
    float logp;
    float base_scale;
};

__device__ __forceinline__ PufCatUniformMix puf_cat_uniform_mix(
        float base_logp, float eps, float inv_legal) {
    if (eps <= 0.0f) {
        return {base_logp, 1.0f};
    }
    float base_prob = expf(base_logp);
    float weighted_base = (1.0f - eps) * base_prob;
    float mixed_prob = weighted_base + eps * inv_legal;
    return {logf(mixed_prob), weighted_base / mixed_prob};
}

// Per-category log probabilities: legal exact, masked literal +0.0f.
__device__ void puf_cat_write_logps(
        const precision_t* __restrict__ logits,
        const precision_t* __restrict__ mask,
        int width, const PufCatNorm& n, float* __restrict__ out) {
    const int lane = puf_cat_lane();
    for (int base = 0; base < width; base += PUF_CAT_WARP) {
        int a = base + lane;
        if (a < width) {
            out[a] = puf_cat_legal(mask[a])
                ? puf_cat_logp(to_float(logits[a]), n) : 0.0f;
        }
    }
}

// Entropy from the cached log probabilities. Each lane only re-reads the
// categories it wrote itself, so no cross-lane visibility is required and the
// cache and gradient stages produce bitwise identical entropy.
__device__ float puf_cat_entropy_from_logps(
        const float* __restrict__ logps, const precision_t* __restrict__ mask,
        int width) {
    const int lane = puf_cat_lane();
    float acc = 0.0f;
    for (int base = 0; base < width; base += PUF_CAT_WARP) {
        int a = base + lane;
        if (a < width && puf_cat_legal(mask[a])) {
            float lp = logps[a];
            float p = expf(lp);
            if (p != 0.0f) {
                acc -= p * lp;
            }
        }
    }
    return puf_cat_warp_sum(acc);
}

// Categorical policy gradient, written in place over the cached logps.
// Masked categories get bitwise +0.0f regardless of their raw logit.
__device__ void puf_cat_write_grads(
        float* __restrict__ grad, const precision_t* __restrict__ mask,
        int width, int action, float d_logp, float d_entropy_term,
        float entropy) {
    const int lane = puf_cat_lane();
    for (int base = 0; base < width; base += PUF_CAT_WARP) {
        int a = base + lane;
        if (a >= width) {
            continue;
        }
        if (!puf_cat_legal(mask[a])) {
            grad[a] = 0.0f;
            continue;
        }
        float lp = grad[a];
        float p = expf(lp);
        // p == 0 (underflowed legal category) contributes no entropy gradient;
        // the guard also keeps 0 * -inf out of the result.
        float ent_term = (p == 0.0f)
            ? 0.0f : d_entropy_term * p * (-entropy - lp);
        grad[a] = ((a == action ? 1.0f : 0.0f) - p) * d_logp + ent_term;
    }
}

__device__ void puf_cat_zero_head(float* __restrict__ grad, int width) {
    const int lane = puf_cat_lane();
    for (int base = 0; base < width; base += PUF_CAT_WARP) {
        int a = base + lane;
        if (a < width) {
            grad[a] = 0.0f;
        }
    }
}

// Inverse-CDF sample over the legal-only distribution, optionally mixed with
// eps * uniform-over-legal (caller supplies eps and 1/legal_count; eps == 0 is
// the plain categorical). Tile-ordered warp prefix scan: a category is only
// selected when its own probability mass pushes the CDF past u, so a masked
// (exactly zero) category can never be chosen. u in (0, 1]; a rounding
// fall-through deterministically lands on the last legal category.
__device__ int puf_cat_sample(
        const precision_t* __restrict__ logits,
        const precision_t* __restrict__ mask,
        int width, const PufCatNorm& n, float u, float eps, float inv_legal) {
    const int lane = puf_cat_lane();
    float carry = 0.0f;
    int chosen = -1;
    for (int base = 0; base < width && chosen < 0; base += PUF_CAT_WARP) {
        int a = base + lane;
        bool legal = (a < width) && puf_cat_legal(mask[a]);
        float p = 0.0f;
        if (legal) {
            p = expf(puf_cat_logp(to_float(logits[a]), n));
            if (eps > 0.0f) {
                p = (1.0f - eps) * p + eps * inv_legal;
            }
        }
        float incl = puf_cat_warp_prefix_incl(p);
        unsigned hit = __ballot_sync(PUF_CAT_ALL_LANES,
            (a < width) && (carry + incl > u));
        if (hit != 0u) {
            chosen = base + (__ffs((int)hit) - 1);
        }
        // Lane 31's inclusive prefix is this tile's total mass.
        carry += __shfl_sync(PUF_CAT_ALL_LANES, incl, PUF_CAT_WARP - 1);
    }
    return chosen < 0 ? n.last_legal : chosen;
}

// Validate a stored float32 action for a head of the given width.
// Returns the category index, or -1 with *out_code set.
__device__ __forceinline__ int puf_cat_check_action(
        float action, int width, int* out_code) {
    if (!isfinite(action) || action != truncf(action)
            || action < 0.0f || action >= (float)width) {
        *out_code = PUF_CAT_ERR_ACTION;
        return -1;
    }
    *out_code = PUF_CAT_OK;
    return (int)action;
}

// Host side: status lifecycle + fail-before-consume check.

// Paired device buffer + pinned host mirror for one stream of work.
struct PufCatStatusPair {
    PufCatStatus* device;
    PufCatStatus* host;
};

void puf_cat_status_reset(PufCatStatus* device, cudaStream_t stream) {
    if (device != NULL) {
        cudaMemsetAsync(device, 0, sizeof(PufCatStatus), stream);
    }
}

void puf_cat_status_fetch(PufCatStatus* host, const PufCatStatus* device,
        cudaStream_t stream) {
    if (host != NULL && device != NULL) {
        cudaMemcpyAsync(host, device, sizeof(PufCatStatus),
            cudaMemcpyDeviceToHost, stream);
    }
}

// Abort before the failing row can reach an environment step, GAE, the loss
// reduction, backward, or the optimizer. Callers must have synchronized the
// stream that produced `host`.
void puf_cat_status_check(const PufCatStatus* host, const char* where) {
    if (host == NULL || host->code == PUF_CAT_OK) {
        return;
    }
    fprintf(stderr,
        "\n[puffer] fatal action-mask error in %s: %s "
        "(code %d, row %d, head %d, detail %d)\n",
        where, puf_cat_code_name(host->code), host->code,
        host->row, host->head, host->detail);
    fflush(stderr);
    abort();
}

// Allocate one status pair (device payload + pinned host mirror).
PufCatStatusPair puf_cat_status_create() {
    PufCatStatusPair p = {NULL, NULL};
    cudaMalloc((void**)&p.device, sizeof(PufCatStatus));
    cudaMemset(p.device, 0, sizeof(PufCatStatus));
    cudaHostAlloc((void**)&p.host, sizeof(PufCatStatus), cudaHostAllocPortable);
    memset(p.host, 0, sizeof(PufCatStatus));
    return p;
}

#endif  // PUF_CATEGORICAL_CUH
