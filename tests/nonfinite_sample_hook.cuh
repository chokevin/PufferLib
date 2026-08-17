// Custom sampler fixture that reports a nonfinite legacy logprob. The runtime
// must surface the converted status before either the environment or PPO can
// consume the row, including continuous-action graph paths.
#ifndef PUF_TEST_NONFINITE_SAMPLE_HOOK_CUH
#define PUF_TEST_NONFINITE_SAMPLE_HOOK_CUH

__global__ void nonfinite_sample_logits(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dec_out.shape[0]) {
        return;
    }
    (void)logstd;
    (void)act_sizes;
    (void)rng_states;
    (void)action_mask;
    (void)mask_stride;
    actions[idx * NUM_ATNS] = 0.0f;
    env_actions[idx * NUM_ATNS] = 0.0f;
    logprobs[idx] = from_float(NAN);
    value_out[idx] = dec_out.data[
        idx * dec_out.shape[1] + dec_out.shape[1] - 1];
}

#endif
