// Deterministic categorical CUDA/reference fixtures.
// Tau build/run:
//   nvcc -O3 -std=c++17 --fmad=false -Xcompiler=-ffp-contract=off \
//     tests/test_wide_categorical.cu -o /tmp/test_wide_categorical
//   /tmp/test_wide_categorical

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

typedef float precision_t;
#define to_float(x) (x)
#include "../src/categorical.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(error_)); \
        exit(1); \
    } \
} while (0)

constexpr int SCALAR_N = 7;

template<int A>
__global__ void categorical_fixture(
        const float* logits, const float* mask, float action,
        float old_logp, float d_logp, float d_entropy,
        float cdf_value, float* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    float lse;
    if constexpr (A <= 32) {
        float cache[A];
        lse = ppo_discrete_logsumexp(logits, 0, 0, A, mask, 0, cache);
        for (int a = 0; a < A; ++a) {
            out[SCALAR_N + a] = cache[a] - lse;
        }
    } else {
        lse = ppo_discrete_logsumexp(logits, 0, 0, A, mask, 0, nullptr);
        for (int a = 0; a < A; ++a) {
            out[SCALAR_N + a] = ppo_discrete_mask_allows(mask, 0, 0, a)
                ? ppo_discrete_load_logit(logits, 0, 0, a) - lse
                : -INFINITY;
        }
    }

    bool valid = ppo_discrete_action_valid(action, A, mask, 0, 0);
    out[0] = lse;
    out[4] = valid ? 1.0f : 0.0f;
    if (!valid) {
        return;
    }

    int selected = (int)action;
    float selected_logp = out[SCALAR_N + selected];
    float entropy = 0.0f;
    float cdf = 0.0f;
    int sampled = -1;
    for (int a = 0; a < A; ++a) {
        int logp_idx = SCALAR_N + a;
        int prob_idx = SCALAR_N + A + a;
        int grad_idx = SCALAR_N + 2 * A + a;
        if (!ppo_discrete_mask_allows(mask, 0, 0, a)) {
            out[prob_idx] = 0.0f;
            out[grad_idx] = 0.0f;
            continue;
        }

        float logp = out[logp_idx];
        float probability = __expf(logp);
        out[prob_idx] = probability;
        entropy -= probability * logp;
        cdf += probability;
        if (sampled < 0 && cdf_value < cdf) {
            sampled = a;
        }
    }
    if (sampled < 0) {
        for (int a = A - 1; a >= 0; --a) {
            if (ppo_discrete_mask_allows(mask, 0, 0, a)) {
                sampled = a;
                break;
            }
        }
    }

    for (int a = 0; a < A; ++a) {
        if (!ppo_discrete_mask_allows(mask, 0, 0, a)) {
            continue;
        }
        float logp = out[SCALAR_N + a];
        float probability = out[SCALAR_N + A + a];
        out[SCALAR_N + 2 * A + a] =
            ((a == selected ? 1.0f : 0.0f) - probability) * d_logp
            + d_entropy * probability * (-entropy - logp);
    }

    out[1] = selected_logp;
    out[2] = entropy;
    out[3] = __expf(selected_logp - old_logp);
    out[5] = (float)sampled;
    out[6] = out[SCALAR_N + sampled];
}

template<int A>
__global__ void selected_action_validity(
        const float* mask, int legal, int masked, int* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    float probes[5] = {
        (float)legal, -1.0f, (float)A, (float)legal + 0.5f, (float)masked,
    };
    for (int i = 0; i < 5; ++i) {
        out[i] = ppo_discrete_action_valid(probes[i], A, mask, 0, 0);
    }
}

struct Reference {
    double lse;
    double selected_logp;
    double entropy;
    double ratio;
    int sampled;
    std::vector<double> logps;
    std::vector<double> probabilities;
    std::vector<double> gradients;
};

static Reference reference(
        const std::vector<float>& logits, const std::vector<float>& mask,
        int selected, double old_logp, double d_logp, double d_entropy,
        double cdf_value) {
    int A = (int)logits.size();
    Reference r = {};
    r.logps.assign(A, -std::numeric_limits<double>::infinity());
    r.probabilities.assign(A, 0.0);
    r.gradients.assign(A, 0.0);

    double max_logit = -std::numeric_limits<double>::infinity();
    for (int a = 0; a < A; ++a) {
        if (mask[a] != 0.0f) {
            max_logit = std::max(max_logit, (double)logits[a]);
        }
    }
    double sum = 0.0;
    for (int a = 0; a < A; ++a) {
        if (mask[a] != 0.0f) {
            sum += exp((double)logits[a] - max_logit);
        }
    }
    r.lse = max_logit + log(sum);

    double cdf = 0.0;
    r.sampled = -1;
    for (int a = 0; a < A; ++a) {
        if (mask[a] == 0.0f) {
            continue;
        }
        r.logps[a] = (double)logits[a] - r.lse;
        r.probabilities[a] = exp(r.logps[a]);
        r.entropy -= r.probabilities[a] * r.logps[a];
        cdf += r.probabilities[a];
        if (r.sampled < 0 && cdf_value < cdf) {
            r.sampled = a;
        }
    }
    if (r.sampled < 0) {
        for (int a = A - 1; a >= 0; --a) {
            if (mask[a] != 0.0f) {
                r.sampled = a;
                break;
            }
        }
    }

    r.selected_logp = r.logps[selected];
    r.ratio = exp(r.selected_logp - old_logp);
    for (int a = 0; a < A; ++a) {
        if (mask[a] == 0.0f) {
            continue;
        }
        double probability = r.probabilities[a];
        r.gradients[a] =
            ((a == selected ? 1.0 : 0.0) - probability) * d_logp
            + d_entropy * probability * (-r.entropy - r.logps[a]);
    }
    return r;
}

static bool close_enough(float actual, double expected, double atol, double rtol) {
    return fabs((double)actual - expected) <= atol + rtol * fabs(expected);
}

template<int A>
static int run_case(const char* name, int mode) {
    std::vector<float> logits(A), mask(A, 1.0f);
    for (int a = 0; a < A; ++a) {
        logits[a] = (float)(((a * 37) % 101) - 50) * 0.125f;
    }

    int selected = A / 2;
    if (mode == 1) {
        for (int a = 0; a < A; ++a) {
            mask[a] = (a % 11 == 3) ? 1.0f : 0.0f;
            logits[a] = mask[a] ? (float)(a % 13) - 6.0f : 5000.0f + a;
        }
        selected = 3;
    } else if (mode == 2) {
        std::fill(mask.begin(), mask.end(), 0.0f);
        selected = A - 2;
        mask[selected] = 1.0f;
        for (int a = 0; a < A; ++a) {
            logits[a] = a == selected ? -700.0f : 700.0f + a;
        }
    } else if (mode == 3) {
        for (int a = 0; a < A; ++a) {
            mask[a] = (a % 5 != 0) ? 1.0f : 0.0f;
            logits[a] = (a == A - 1) ? 1000.0f : -1000.0f + (float)(a % 17);
        }
        selected = (mask[A - 1] != 0.0f) ? A - 1 : A - 2;
    }

    constexpr double d_logp = 0.37;
    constexpr double d_entropy = -0.02;
    constexpr double cdf_value = 0.371;
    Reference pre = reference(
        logits, mask, selected, 0.0, d_logp, d_entropy, cdf_value);
    double old_logp = pre.selected_logp + 0.125;
    Reference expected = reference(
        logits, mask, selected, old_logp, d_logp, d_entropy, cdf_value);

    float *d_logits, *d_mask, *d_out;
    size_t vector_bytes = (size_t)A * sizeof(float);
    size_t out_count = SCALAR_N + 3 * A;
    CUDA_CHECK(cudaMalloc(&d_logits, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_mask, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        d_logits, logits.data(), vector_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_mask, mask.data(), vector_bytes, cudaMemcpyHostToDevice));

    categorical_fixture<A><<<1, 1>>>(
        d_logits, d_mask, (float)selected, (float)old_logp,
        (float)d_logp, (float)d_entropy, (float)cdf_value, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(out_count), repeated(out_count);
    CUDA_CHECK(cudaMemcpy(
        actual.data(), d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost));
    categorical_fixture<A><<<1, 1>>>(
        d_logits, d_mask, (float)selected, (float)old_logp,
        (float)d_logp, (float)d_entropy, (float)cdf_value, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        repeated.data(), d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost));

    int fail = 0;
    if (memcmp(actual.data(), repeated.data(), out_count * sizeof(float)) != 0) {
        fprintf(stderr, "FAIL A=%d %s: non-deterministic output\n", A, name);
        fail++;
    }
    if (actual[4] != 1.0f
            || !close_enough(actual[0], expected.lse, 2e-4, 2e-6)
            || !close_enough(actual[1], expected.selected_logp, 3e-4, 2e-6)
            || !close_enough(actual[2], expected.entropy, 3e-4, 3e-5)
            || !close_enough(actual[3], expected.ratio, 3e-5, 3e-5)
            || (int)actual[5] != expected.sampled
            || !close_enough(
                actual[6], expected.logps[expected.sampled], 3e-4, 2e-6)) {
        fprintf(stderr, "FAIL A=%d %s: scalar/reference mismatch\n", A, name);
        fail++;
    }
    for (int a = 0; a < A; ++a) {
        float logp = actual[SCALAR_N + a];
        float probability = actual[SCALAR_N + A + a];
        float gradient = actual[SCALAR_N + 2 * A + a];
        if (mask[a] == 0.0f) {
            if (!isinf(logp) || logp >= 0.0f
                    || probability != 0.0f || gradient != 0.0f) {
                fprintf(stderr,
                    "FAIL A=%d %s: masked category %d was not exact zero\n",
                    A, name, a);
                fail++;
                break;
            }
        } else if (!close_enough(logp, expected.logps[a], 3e-4, 2e-6)
                || !close_enough(
                    probability, expected.probabilities[a], 2e-5, 3e-5)
                || !close_enough(
                    gradient, expected.gradients[a], 3e-5, 5e-5)) {
            fprintf(stderr,
                "FAIL A=%d %s: category %d reference mismatch\n",
                A, name, a);
            fail++;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_out));
    return fail;
}

template<int A>
static int run_validity() {
    std::vector<float> mask(A, 1.0f);
    int legal = A / 2;
    int masked = (legal + 1) % A;
    mask[masked] = 0.0f;
    float* d_mask;
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_mask, (size_t)A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, 5 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data(),
        (size_t)A * sizeof(float), cudaMemcpyHostToDevice));
    selected_action_validity<A><<<1, 1>>>(d_mask, legal, masked, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    int actual[5] = {};
    CUDA_CHECK(cudaMemcpy(
        actual, d_out, sizeof(actual), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_out));
    const int expected[5] = {1, 0, 0, 0, 0};
    if (memcmp(actual, expected, sizeof(actual)) != 0) {
        fprintf(stderr, "FAIL A=%d selected-action validity\n", A);
        return 1;
    }
    return 0;
}

template<int A>
static int run_width() {
    int fail = 0;
    fail += run_case<A>("dense", 0);
    fail += run_case<A>("sparse", 1);
    fail += run_case<A>("singleton", 2);
    fail += run_case<A>("extreme", 3);
    fail += run_validity<A>();
    printf("A=%d %s\n", A, fail == 0 ? "ok" : "FAIL");
    return fail;
}

int main() {
    int fail = 0;
    fail += run_width<31>();
    fail += run_width<32>();
    fail += run_width<33>();
    fail += run_width<155>();
    fail += run_width<212>();
    printf("fail=%d\n", fail);
    return fail == 0 ? 0 : 1;
}
