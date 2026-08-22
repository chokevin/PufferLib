#pragma once

// Requires precision_t and to_float from the including translation unit.

__device__ __forceinline__ bool ppo_discrete_mask_allows(
        const precision_t* __restrict__ mask, int mask_base,
        int logits_offset, int a) {
    return to_float(mask[mask_base + logits_offset + a]) != 0.0f;
}

__device__ __forceinline__ float ppo_discrete_load_logit(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_offset, int a) {
    return to_float(logits[logits_base + logits_offset + a]);
}

// Stable online log-sum-exp over legal categories only. cache may be null for
// the wide specialization; masked cache entries are exact log(0).
__device__ __forceinline__ float ppo_discrete_logsumexp(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_offset, int A,
        const precision_t* __restrict__ mask, int mask_base,
        float* __restrict__ cache) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    int legal = 0;
    for (int a = 0; a < A; ++a) {
        if (!ppo_discrete_mask_allows(mask, mask_base, logits_offset, a)) {
            if (cache != nullptr) {
                cache[a] = -INFINITY;
            }
            continue;
        }

        float l = ppo_discrete_load_logit(logits, logits_base, logits_offset, a);
        if (cache != nullptr) {
            cache[a] = l;
        }
        if (legal == 0) {
            max_logit = l;
            sum = 1.0f;
        } else if (l > max_logit) {
            sum = sum * __expf(max_logit - l) + 1.0f;
            max_logit = l;
        } else {
            sum += __expf(l - max_logit);
        }
        legal++;
    }

    assert(legal > 0 && "categorical action head has no legal actions");
    return max_logit + __logf(sum);
}

__device__ __forceinline__ bool ppo_discrete_action_valid(
        float action, int A,
        const precision_t* __restrict__ mask, int mask_base,
        int logits_offset) {
    if (!isfinite(action) || action < 0.0f || action >= (float)A) {
        return false;
    }
    int act = (int)action;
    return action == (float)act
        && ppo_discrete_mask_allows(mask, mask_base, logits_offset, act);
}

__device__ __forceinline__ int ppo_discrete_validate_action(
        float action, int A,
        const precision_t* __restrict__ mask, int mask_base,
        int logits_offset) {
    bool valid = ppo_discrete_action_valid(
        action, A, mask, mask_base, logits_offset);
    assert(valid && "selected categorical action is invalid or masked");
    return valid ? (int)action : 0;
}
