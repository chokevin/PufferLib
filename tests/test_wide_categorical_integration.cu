// Production-kernel integration for mixed narrow/wide categorical heads.
// Tau build/run (same CUDA/library inputs as build.sh native mode):
//   nvcc -O2 -std=c++17 --fmad=false -Xptxas=-v \
//     -Xcompiler=-ffp-contract=off \
//     -DPRECISION_FLOAT -DPLATFORM_DESKTOP \
//     -DENV_HEADER=\"tests/wide_categorical_env.h\" \
//     -DENV_NAME=wide_categorical_test \
//     -DPUFFER_ENV_NAME=\"wide_categorical_test\" \
//     -I. -Isrc -Itests -Ivendor $CUDA_NCCL_RAYLIB_INCLUDES \
//     tests/test_wide_categorical_integration.cu $CUDA_NCCL_RAYLIB_LIBS \
//     -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
//     -lm -lpthread -o /tmp/test_wide_categorical_integration
//   /tmp/test_wide_categorical_integration
// Repeat with -DTEST_NARROW_CATEGORICAL to exercise the cached production
// specialization at widths 31 and 32.

#include "../src/pufferl.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(error_)); \
        exit(1); \
    } \
} while (0)

#ifdef TEST_NARROW_CATEGORICAL
constexpr int TEST_HEADS = 2;
constexpr int TEST_SIZES[TEST_HEADS] = {31, 32};
constexpr int TEST_A_TOTAL = 31 + 32;
#else
constexpr int TEST_HEADS = 5;
constexpr int TEST_SIZES[TEST_HEADS] = {31, 32, 33, 155, 212};
constexpr int TEST_A_TOTAL = 31 + 32 + 33 + 155 + 212;
#endif

struct HeadRef {
    double lse;
    double entropy;
    std::vector<double> logps;
    std::vector<double> probs;
};

static HeadRef head_reference(
        const std::vector<float>& logits, const std::vector<float>& mask,
        int offset, int A) {
    HeadRef r = {};
    r.logps.assign(A, -std::numeric_limits<double>::infinity());
    r.probs.assign(A, 0.0);
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int a = 0; a < A; ++a) {
        if (mask[offset + a] != 0.0f) {
            max_logit = std::max(max_logit, (double)logits[offset + a]);
        }
    }
    double sum = 0.0;
    for (int a = 0; a < A; ++a) {
        if (mask[offset + a] != 0.0f) {
            sum += exp((double)logits[offset + a] - max_logit);
        }
    }
    r.lse = max_logit + log(sum);
    for (int a = 0; a < A; ++a) {
        if (mask[offset + a] == 0.0f) {
            continue;
        }
        r.logps[a] = (double)logits[offset + a] - r.lse;
        r.probs[a] = exp(r.logps[a]);
        r.entropy -= r.probs[a] * r.logps[a];
    }
    return r;
}

static bool close_enough(float actual, double expected, double atol, double rtol) {
    return fabs((double)actual - expected) <= atol + rtol * fabs(expected);
}

int main() {
    static_assert(!USE_BF16, "compile this fixture with -DPRECISION_FLOAT");
#ifdef TEST_NARROW_CATEGORICAL
    static_assert(!PPO_WIDE_CATEGORICAL, "fixture must select cached kernels");
    static_assert(PPO_MAX_HEAD_A == 32, "fixture must exercise width 32");
#else
    static_assert(PPO_WIDE_CATEGORICAL, "mixed fixture must select wide kernels");
    static_assert(PPO_MAX_HEAD_A == 212, "fixture must exercise width 212");
#endif

    std::vector<float> logits(TEST_A_TOTAL + 1), mask(TEST_A_TOTAL, 1.0f);
    std::vector<float> actions(TEST_HEADS);
    int offset = 0;
    for (int h = 0; h < TEST_HEADS; ++h) {
        int A = TEST_SIZES[h];
        for (int a = 0; a < A; ++a) {
            logits[offset + a] =
                (float)(((a * 37 + h * 11) % 101) - 50) * 0.125f;
        }
        actions[h] = (float)(A / 2);
        offset += A;
    }
    logits[TEST_A_TOTAL] = 0.75f;

    offset = 0;
    for (int h = 0; h < TEST_HEADS; ++h) {
        int A = TEST_SIZES[h];
        if (h == 1) {
            for (int a = 0; a < A; ++a) {
                mask[offset + a] = (a % 7 == 2) ? 1.0f : 0.0f;
                logits[offset + a] =
                    mask[offset + a] ? (float)(a % 9) : 5000.0f + a;
            }
            actions[h] = 2.0f;
        } else if (h == 2) {
            std::fill(mask.begin() + offset, mask.begin() + offset + A, 0.0f);
            actions[h] = 17.0f;
            mask[offset + 17] = 1.0f;
            for (int a = 0; a < A; ++a) {
                logits[offset + a] = a == 17 ? -700.0f : 700.0f + a;
            }
        } else if (h == 3) {
            for (int a = 0; a < A; ++a) {
                mask[offset + a] = (a % 5 != 0) ? 1.0f : 0.0f;
                logits[offset + a] =
                    a == A - 1 ? 1000.0f : -1000.0f + (a % 17);
            }
            actions[h] = (float)(A - 1);
        } else if (h == 4) {
            for (int a = 0; a < A; ++a) {
                mask[offset + a] = (a % 17 == 4) ? 1.0f : 0.0f;
                logits[offset + a] =
                    mask[offset + a] ? (float)(a % 23) - 11.0f : 9000.0f + a;
            }
            actions[h] = 4.0f;
        }
        offset += A;
    }

    std::vector<HeadRef> refs;
    double new_logp_ref = 0.0;
    double entropy_ref = 0.0;
    offset = 0;
    for (int h = 0; h < TEST_HEADS; ++h) {
        refs.push_back(head_reference(logits, mask, offset, TEST_SIZES[h]));
        new_logp_ref += refs.back().logps[(int)actions[h]];
        entropy_ref += refs.back().entropy;
        offset += TEST_SIZES[h];
    }
    float old_logp = (float)(new_logp_ref + 0.05);

    float *d_logits, *d_actions, *d_mask, *d_logps, *d_new_logp;
    float *d_imp, *d_value, *d_env_actions, *d_sample_logp, *d_sample_value;
    int* d_sizes;
    curandStatePhilox4_32_10_t* d_rng;
    CUDA_CHECK(cudaMalloc(&d_logits, logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_actions, actions.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, mask.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logps, TEST_A_TOTAL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_logp, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_imp, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_value, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_env_actions, TEST_HEADS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sample_logp, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sample_value, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sizes, TEST_HEADS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rng, sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(),
        logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_actions, actions.data(),
        actions.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data(),
        mask.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, TEST_SIZES,
        sizeof(TEST_SIZES), cudaMemcpyHostToDevice));
    float* d_old_logp;
    CUDA_CHECK(cudaMalloc(&d_old_logp, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        d_old_logp, &old_logp, sizeof(float), cudaMemcpyHostToDevice));

    Prec train_dec = {.data = d_logits, .shape = {1, 1, TEST_A_TOTAL + 1}};
    Prec empty = {};
    cache_imp_and_v<<<1, 1>>>(
        train_dec, d_actions, d_old_logp, d_mask, empty, d_sizes,
        d_imp, d_value, d_logps, d_new_logp);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual_logps(TEST_A_TOTAL);
    float actual_new_logp, actual_imp, actual_value;
    CUDA_CHECK(cudaMemcpy(actual_logps.data(), d_logps,
        TEST_A_TOTAL * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &actual_new_logp, d_new_logp, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &actual_imp, d_imp, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &actual_value, d_value, sizeof(float), cudaMemcpyDeviceToHost));

    int fail = 0;
    if (!close_enough(actual_new_logp, new_logp_ref, 8e-4, 3e-6)
            || !close_enough(
                actual_imp, exp(new_logp_ref - old_logp), 4e-5, 4e-5)
            || actual_value != logits[TEST_A_TOTAL]) {
        fprintf(stderr, "FAIL production PPO forward scalars\n");
        fail++;
    }
    offset = 0;
    for (int h = 0; h < TEST_HEADS; ++h) {
        for (int a = 0; a < TEST_SIZES[h]; ++a) {
            float actual = actual_logps[offset + a];
            if (mask[offset + a] == 0.0f) {
                if (!isinf(actual) || actual >= 0.0f) {
                    fprintf(stderr, "FAIL masked global logp h=%d a=%d\n", h, a);
                    fail++;
                }
            } else if (!close_enough(
                    actual, refs[h].logps[a], 4e-4, 3e-6)) {
                fprintf(stderr, "FAIL global logp h=%d a=%d\n", h, a);
                fail++;
            }
        }
        offset += TEST_SIZES[h];
    }

    float advantage = 0.7f, value = 0.6f, ret = 0.4f, ent_coef = 0.01f;
    float *d_advantage, *d_old_value, *d_return, *d_ent_coef, *d_partials;
    CUDA_CHECK(cudaMalloc(&d_advantage, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_old_value, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_return, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ent_coef, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partials, LOSS_N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        d_advantage, &advantage, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_old_value, &value, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_return, &ret, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_ent_coef, &ent_coef, sizeof(float), cudaMemcpyHostToDevice));

    PPOGraphArgs graph = {
        .imp = d_imp,
        .actions = d_actions,
        .old_logprobs = d_old_logp,
        .advantages = d_advantage,
        .values = d_old_value,
        .returns = d_return,
    };
    PPOKernelArgs args = {
        .grad_logits = d_logps,
        .grad_logstd = nullptr,
        .grad_values_pred = d_new_logp,
        .logits = d_logits,
        .logstd = nullptr,
        .values_pred = d_logits + TEST_A_TOTAL,
        .act_sizes = d_sizes,
        .action_mask = d_mask,
        .num_atns = TEST_HEADS,
        .clip_coef = 0.2f,
        .vf_clip_coef = 0.2f,
        .vf_coef = 0.5f,
        .ent_coef = d_ent_coef,
        .T_seq = 1,
        .A_total = TEST_A_TOTAL,
        .N = 1,
        .is_continuous = false,
    };
    ppo_loss_compute<<<1, PPO_THREADS>>>(d_partials, args, graph);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> gradients(TEST_A_TOTAL), partials(LOSS_N);
    CUDA_CHECK(cudaMemcpy(gradients.data(), d_logps,
        TEST_A_TOTAL * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(partials.data(), d_partials,
        LOSS_N * sizeof(float), cudaMemcpyDeviceToHost));

    double ratio = exp(new_logp_ref - old_logp);
    double expected_d_new_logp = -(double)advantage * ratio;
    offset = 0;
    for (int h = 0; h < TEST_HEADS; ++h) {
        int selected = (int)actions[h];
        for (int a = 0; a < TEST_SIZES[h]; ++a) {
            double expected = 0.0;
            if (mask[offset + a] != 0.0f) {
                double p = refs[h].probs[a];
                expected = ((a == selected ? 1.0 : 0.0) - p)
                    * expected_d_new_logp
                    - ent_coef * p * (-refs[h].entropy - refs[h].logps[a]);
            }
            if (mask[offset + a] == 0.0f && gradients[offset + a] != 0.0f) {
                fprintf(stderr, "FAIL masked gradient h=%d a=%d\n", h, a);
                fail++;
            } else if (!close_enough(
                    gradients[offset + a], expected, 5e-4, 8e-5)) {
                fprintf(stderr, "FAIL gradient h=%d a=%d\n", h, a);
                fail++;
            }
        }
        offset += TEST_SIZES[h];
    }
    if (!close_enough(partials[LOSS_ENT], entropy_ref, 5e-4, 5e-5)) {
        fprintf(stderr, "FAIL production entropy\n");
        fail++;
    }

    Prec sample_dec = {.data = d_logits, .shape = {1, TEST_A_TOTAL + 1}};
    float sampled_actions[TEST_HEADS], repeated_actions[TEST_HEADS];
    float sampled_logp, repeated_logp;
    rng_init<<<1, 1>>>(d_rng, 1234, 1);
    sample_logits<<<1, 1>>>(
        sample_dec, empty, d_sizes, d_actions, d_env_actions,
        d_sample_logp, d_sample_value, d_rng, d_mask, TEST_A_TOTAL);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(sampled_actions, d_actions,
        sizeof(sampled_actions), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &sampled_logp, d_sample_logp, sizeof(float), cudaMemcpyDeviceToHost));
    rng_init<<<1, 1>>>(d_rng, 1234, 1);
    sample_logits<<<1, 1>>>(
        sample_dec, empty, d_sizes, d_actions, d_env_actions,
        d_sample_logp, d_sample_value, d_rng, d_mask, TEST_A_TOTAL);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(repeated_actions, d_actions,
        sizeof(repeated_actions), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &repeated_logp, d_sample_logp, sizeof(float), cudaMemcpyDeviceToHost));
    if (memcmp(sampled_actions, repeated_actions, sizeof(sampled_actions)) != 0
            || sampled_logp != repeated_logp) {
        fprintf(stderr, "FAIL production sampler determinism\n");
        fail++;
    }
    double sampled_logp_ref = 0.0;
    offset = 0;
    for (int h = 0; h < TEST_HEADS; ++h) {
        bool in_range = isfinite(sampled_actions[h])
            && sampled_actions[h] >= 0.0f
            && sampled_actions[h] < (float)TEST_SIZES[h];
        int sampled = in_range ? (int)sampled_actions[h] : -1;
        bool valid = in_range && sampled_actions[h] == (float)sampled
            && mask[offset + sampled] != 0.0f;
        if (!valid) {
            fprintf(stderr, "FAIL production sampler validity h=%d\n", h);
            fail++;
        } else {
            sampled_logp_ref += refs[h].logps[sampled];
        }
        offset += TEST_SIZES[h];
    }
    if (!close_enough(sampled_logp, sampled_logp_ref, 8e-4, 3e-6)) {
        fprintf(stderr, "FAIL production sampler logprob\n");
        fail++;
    }

    cudaFree(d_logits);
    cudaFree(d_actions);
    cudaFree(d_mask);
    cudaFree(d_logps);
    cudaFree(d_new_logp);
    cudaFree(d_imp);
    cudaFree(d_value);
    cudaFree(d_env_actions);
    cudaFree(d_sample_logp);
    cudaFree(d_sample_value);
    cudaFree(d_sizes);
    cudaFree(d_rng);
    cudaFree(d_old_logp);
    cudaFree(d_advantage);
    cudaFree(d_old_value);
    cudaFree(d_return);
    cudaFree(d_ent_coef);
    cudaFree(d_partials);
    printf("wide categorical production integration: %s\n",
        fail == 0 ? "ok" : "FAIL");
    return fail == 0 ? 0 : 1;
}
