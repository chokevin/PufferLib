// Host reference shim: compiles src/ppo_group.h (the exact header the CUDA
// kernels use) into a shared library so tests/test_ppo_group.py can exercise
// the grouped-PPO config validation and objective math without a GPU.
#include <string.h>

#include "../src/ppo_group.h"

extern "C" {

int ppo_shim_clip_mode_parse(const char* raw) {
    return ppo_clip_mode_parse(raw);
}

int ppo_shim_group_ids_is_disabled(const char* raw) {
    return ppo_group_ids_is_disabled(raw);
}

int ppo_shim_config_validate(const char* mode_raw, const char* ids_raw,
        int num_atns, const int* act_sizes, int is_continuous, int vtrace,
        int max_head_classes, int* mode_out, int* ids, int* num_groups,
        char* err, int err_n) {
    return ppo_group_config_validate(mode_raw, ids_raw, num_atns, act_sizes,
        is_continuous, vtrace, max_head_classes, mode_out, ids, num_groups,
        err, err_n);
}

// Thin wrapper over the row objective. Mirrors PpoGroupStats field order.
int ppo_shim_group_row(int mode, int num_groups, const float* logratio_sum,
        const int* active_count, float adv, float clip_coef, float inv_nt,
        float* dlogp_out, float* stats_out, int* active_groups_out) {
    PpoGroupStats st;
    int code = ppo_group_row(mode, num_groups, logratio_sum, active_count, adv,
        clip_coef, inv_nt, dlogp_out, &st);
    stats_out[0] = st.pg_loss;
    stats_out[1] = st.old_kl;
    stats_out[2] = st.kl;
    stats_out[3] = st.clipfrac;
    stats_out[4] = st.importance;
    stats_out[5] = st.abs_logratio;
    stats_out[6] = st.logratio_max;
    stats_out[7] = st.logratio_min;
    stats_out[8] = st.ratio_max;
    *active_groups_out = st.active_groups;
    return code;
}

int ppo_shim_mask_legal_count(const float* mask_row, int A, int* out) {
    return ppo_mask_legal_count(mask_row, A, out);
}

int ppo_shim_head_active(int consumed, int legal_count) {
    return ppo_head_active(consumed, legal_count);
}

// Full per-transition forward+backward, identical to the code the CUDA kernel
// runs (ppo_group_row_apply). logps_inout carries per-action log-probs in and
// logit gradients out.
int ppo_shim_group_row_apply(int mode, int num_atns, int num_groups,
        const int* act_sizes, const int* group_ids, const int* legal_count,
        const int* consumed, const float* actions, const float* old_group_lp,
        float adv, float clip_coef, float ent_coef, float inv_nt,
        float* logps_inout, float* stats_out, float* entropy_out,
        int* active_heads_out, int* active_groups_out) {
    PpoGroupRowInput in;
    in.num_atns = num_atns;
    in.num_groups = num_groups;
    in.mode = mode;
    in.act_sizes = act_sizes;
    in.group_ids = group_ids;
    in.legal_count = legal_count;
    in.consumed = consumed;
    in.actions = actions;
    in.old_group_lp = old_group_lp;
    in.head_lp_mix = NULL;
    in.head_grad_scale = NULL;
    in.adv = adv;
    in.clip_coef = clip_coef;
    in.ent_coef = ent_coef;
    in.inv_nt = inv_nt;

    PpoGroupStats st;
    int code = ppo_group_row_apply<PPO_MAX_GROUPS>(&in, logps_inout, &st,
        entropy_out, active_heads_out);
    stats_out[0] = st.pg_loss;
    stats_out[1] = st.old_kl;
    stats_out[2] = st.kl;
    stats_out[3] = st.clipfrac;
    stats_out[4] = st.importance;
    stats_out[5] = st.abs_logratio;
    stats_out[6] = st.logratio_max;
    stats_out[7] = st.logratio_min;
    stats_out[8] = st.ratio_max;
    *active_groups_out = st.active_groups;
    return code;
}

int ppo_shim_max_heads(void) {
    return PPO_MAX_HEADS;
}

int ppo_shim_max_groups(void) {
    return PPO_MAX_GROUPS;
}

}  // extern "C"
