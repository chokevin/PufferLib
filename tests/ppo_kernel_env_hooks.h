// Stubs for the env-conditional head-used hooks, so the PUFFER_NETHACK and
// KG_HEAD_GATING_SELECTOR_VALUES branches are type-checked too.
#pragma once
#ifdef PUFFER_NETHACK
static inline int nethack_head_used(int, int) { return 1; }
static inline float nethack_verb_eps_load(const precision_t*, int, float* invk) {
    *invk = 0.0f; return 0.0f;
}
static inline float nethack_verb_eps_mix(float p, precision_t, float, float) {
    return p;
}
static inline float nethack_verb_train_logp(const precision_t*, int, float lp,
        float* scale) { *scale = 1.0f; return lp; }
#endif
#ifdef PUFFER_PROVIDES_PPO_HEAD_USED
static inline int puffer_ppo_head_used(const float*, int) { return 1; }
#endif
#ifdef KG_HEAD_GATING_SELECTOR_VALUES
static inline int kg_puffer5_head_used(const float*, int) { return 1; }
#endif
