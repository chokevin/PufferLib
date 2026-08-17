// Legacy custom-sampler hook used by tests/test_categorical_mask_source.py's
// compile check. It reproduces the historical ENV_SAMPLE_LOGITS contract
// exactly: an environment header sets
//
//   -DENV_ACTION_KERNEL_HEADER='"../tests/legacy_sample_hook.cuh"'
//   -DENV_SAMPLE_LOGITS=legacy_sample_logits
//
// and src/pufferl.cu launches it with the original ten arguments and the
// original <<<n, BLOCK_SIZE>>> geometry. The trainer then converts the
// precision logprobs this hook writes into the float32 PPO shadow.
#ifndef PUF_TEST_LEGACY_SAMPLE_HOOK_CUH
#define PUF_TEST_LEGACY_SAMPLE_HOOK_CUH

// Deliberately simple: greedy over legal categories, one thread per row.
__global__ void legacy_sample_logits(
        Prec dec_out,
        Prec logstd,
        int* act_sizes,
        float* actions,
        float* env_actions,
        precision_t* logprobs,
        precision_t* value_out,
        curandStatePhilox4_32_10_t* rng_states,
        precision_t* action_mask,
        int mask_stride) {
    int B = dec_out.shape[0];
    int fused_cols = dec_out.shape[1];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) {
        return;
    }
    (void)logstd;
    (void)rng_states;
    precision_t* logits = dec_out.data;
    int logits_base = idx * fused_cols;
    int mask_base = idx * mask_stride;
    int offset = 0;
    float total_logp = 0.0f;
    for (int h = 0; h < NUM_ATNS; h++) {
        int A = act_sizes[h];
        int best = 0;
        float best_logit = -INFINITY;
        float max_legal = -INFINITY;
        for (int a = 0; a < A; a++) {
            if (to_float(action_mask[mask_base + offset + a]) == 0.0f) {
                continue;
            }
            float l = to_float(logits[logits_base + offset + a]);
            if (l > best_logit) {
                best_logit = l;
                best = a;
            }
            if (l > max_legal) {
                max_legal = l;
            }
        }
        float sum = 0.0f;
        for (int a = 0; a < A; a++) {
            if (to_float(action_mask[mask_base + offset + a]) != 0.0f) {
                sum += expf(to_float(logits[logits_base + offset + a]) - max_legal);
            }
        }
        actions[idx * NUM_ATNS + h] = (float)best;
        env_actions[idx * NUM_ATNS + h] = (float)best;
        total_logp += (best_logit - max_legal) - logf(sum);
        offset += A;
    }
    logprobs[idx] = from_float(total_logp);
    value_out[idx] = logits[logits_base + fused_cols - 1];
}

#endif  // PUF_TEST_LEGACY_SAMPLE_HOOK_CUH
