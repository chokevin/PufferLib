#pragma once
// Grouped PPO clipping: config parsing/validation + the exact per-transition
// surrogate math. Shared verbatim by the production CUDA kernels (algo.cu) and
// by the host reference tests (tests/ppo_group_shim.c), so the numbers proven
// on CPU are the numbers the kernel computes.
//
// Vocabulary
//   head       one categorical action factor of the env (NUM_ATNS of them)
//   group      set of heads sharing a ppo_group_ids entry; groups never span
//              two original env-agent transitions (ids index heads inside one
//              transition row, so a group is row-local by construction)
//   consumed   env says the head was actually used this step (nethack/KG hooks;
//              generic envs consume every head by definition)
//   active     consumed AND stochastic (legal-mask cardinality > 1)
//
// Conditional-head environment adapter
//   #define PUFFER_PROVIDES_PPO_HEAD_USED 1
//   __device__ int puffer_ppo_head_used(
//       const float* actions_row, int head);
// Generic environments omit the marker and consume every head. The function
// must return exactly 0 or 1 from the stored sampled action row.
//
// train.ppo_clip_mode
//   joint      legacy default: one scalar log-ratio per transition
//   per_head   one head per factor (identity grouping), clip and average
//   group_sum  sum member log-ratios per group, clip each active group,
//              average clipped surrogates over active groups
//   group_mean average member log-ratios inside a group, then clip, then
//              average over active groups

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef __CUDACC__
#define PPO_HD __host__ __device__ __forceinline__
#else
#define PPO_HD static inline
#endif

// CUDA shape bounds. Grouped kernels keep fixed-size local arrays (no VLAs),
// so heads and groups are checked against these at startup.
#define PPO_MAX_HEADS 32
#define PPO_MAX_GROUPS PPO_MAX_HEADS
#define PUFFER_GROUPED_PPO_CAPABILITY "pufferlib.grouped_ppo.v1"
#define PUFFER_GROUPED_PPO_CAPABILITY_VERSION 1

enum PpoClipMode {
    PPO_CLIP_JOINT = 0,
    PPO_CLIP_PER_HEAD = 1,
    PPO_CLIP_GROUP_SUM = 2,
    PPO_CLIP_GROUP_MEAN = 3,
};

// Device error codes. 0 = ok. Fail closed: any nonzero aborts the run at the
// next host synchronization boundary.
enum PpoErr {
    PPO_ERR_NONE = 0,
    PPO_ERR_MASK_NOT_BINARY = 1,
    PPO_ERR_MASK_NO_LEGAL = 2,
    PPO_ERR_ACTION_RANGE = 3,
    PPO_ERR_ACTION_ILLEGAL = 4,
    PPO_ERR_NO_ACTIVE_FACTORS = 5,
    PPO_ERR_NONFINITE = 6,
    PPO_ERR_SHAPE = 7,
};

static inline const char* ppo_err_name(int code) {
    switch (code) {
        case PPO_ERR_NONE: return "ok";
        case PPO_ERR_MASK_NOT_BINARY:
            return "action mask entry is not exactly legal(1)/illegal(0) or is not finite";
        case PPO_ERR_MASK_NO_LEGAL: return "action head has no legal action";
        case PPO_ERR_ACTION_RANGE: return "sampled action index out of range";
        case PPO_ERR_ACTION_ILLEGAL: return "sampled action index is masked illegal";
        case PPO_ERR_NO_ACTIVE_FACTORS:
            return "transition has no active (consumed and stochastic) factor";
        case PPO_ERR_NONFINITE: return "non-finite log-ratio / ratio / loss";
        case PPO_ERR_SHAPE: return "unsupported head/group shape for the CUDA kernels";
        default: return "unknown ppo error";
    }
}

static inline const char* ppo_clip_mode_name(int mode) {
    switch (mode) {
        case PPO_CLIP_JOINT: return "joint";
        case PPO_CLIP_PER_HEAD: return "per_head";
        case PPO_CLIP_GROUP_SUM: return "group_sum";
        case PPO_CLIP_GROUP_MEAN: return "group_mean";
        default: return "invalid";
    }
}

// -1 on unknown mode (fail closed at startup).
static inline int ppo_clip_mode_parse(const char* raw) {
    if (!raw) {
        return -1;
    }
    while (*raw == ' ' || *raw == '\t') {
        raw++;
    }
    size_t n = strlen(raw);
    while (n > 0 && (raw[n - 1] == ' ' || raw[n - 1] == '\t')) {
        n--;
    }
    struct { const char* name; int mode; } table[] = {
        {"joint", PPO_CLIP_JOINT},
        {"per_head", PPO_CLIP_PER_HEAD},
        {"group_sum", PPO_CLIP_GROUP_SUM},
        {"group_mean", PPO_CLIP_GROUP_MEAN},
    };
    for (int i = 0; i < 4; i++) {
        if (strlen(table[i].name) == n && strncmp(raw, table[i].name, n) == 0) {
            return table[i].mode;
        }
    }
    return -1;
}

// The disabled sentinel is the exact literal "-1": unambiguous because every
// valid group id is a nonnegative dense 0-based integer, and it round-trips
// through the existing INI scalar representation.
static inline int ppo_group_ids_is_disabled(const char* raw) {
    if (!raw) {
        return 0;
    }
    while (*raw == ' ' || *raw == '\t') {
        raw++;
    }
    size_t n = strlen(raw);
    while (n > 0 && (raw[n - 1] == ' ' || raw[n - 1] == '\t')) {
        n--;
    }
    return n == 2 && raw[0] == '-' && raw[1] == '1';
}

// Strict dense 0-based CSV mapping. Exactly num_atns tokens, each a plain
// nonnegative decimal integer, and the id set must cover 0..max with no holes.
// Arbitrary non-contiguous head orders are fine ("1,0,1,2" is dense).
// Returns 1 on success, 0 on failure with a message in err.
static inline int ppo_group_ids_parse(const char* raw, int num_atns,
        int* ids, int* num_groups, char* err, int err_n) {
    if (!raw || num_atns <= 0) {
        snprintf(err, err_n, "train.ppo_group_ids: missing mapping");
        return 0;
    }
    if (num_atns > PPO_MAX_HEADS) {
        snprintf(err, err_n,
            "train.ppo_group_ids: %d action heads exceeds PPO_MAX_HEADS=%d",
            num_atns, PPO_MAX_HEADS);
        return 0;
    }

    int n = 0;
    const char* p = raw;
    while (1) {
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        int digits = 0;
        long v = 0;
        while (*p >= '0' && *p <= '9') {
            v = v * 10 + (*p - '0');
            if (v >= PPO_MAX_GROUPS) {
                v = PPO_MAX_GROUPS;  // saturate: anything this large is invalid
            }
            digits++;
            p++;
        }
        if (!digits) {
            snprintf(err, err_n,
                "train.ppo_group_ids: expected a nonnegative integer at "
                "position %d (got \"%s\")", n, raw);
            return 0;
        }
        if (n >= num_atns) {
            snprintf(err, err_n,
                "train.ppo_group_ids: too many entries, expected NUM_ATNS=%d",
                num_atns);
            return 0;
        }
        if (v >= PPO_MAX_GROUPS) {
            snprintf(err, err_n,
                "train.ppo_group_ids: id at position %d exceeds "
                "PPO_MAX_GROUPS=%d", n, PPO_MAX_GROUPS);
            return 0;
        }
        ids[n++] = (int)v;
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == 0) {
            break;
        }
        snprintf(err, err_n,
            "train.ppo_group_ids: unexpected character '%c'; expected a "
            "comma-separated list of nonnegative integers", *p);
        return 0;
    }

    if (n != num_atns) {
        snprintf(err, err_n,
            "train.ppo_group_ids: got %d ids, expected NUM_ATNS=%d", n, num_atns);
        return 0;
    }

    int max_id = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] > max_id) {
            max_id = ids[i];
        }
    }
    int seen[PPO_MAX_GROUPS];
    for (int g = 0; g <= max_id && g < PPO_MAX_GROUPS; g++) {
        seen[g] = 0;
    }
    for (int i = 0; i < n; i++) {
        seen[ids[i]] = 1;
    }
    for (int g = 0; g <= max_id; g++) {
        if (!seen[g]) {
            snprintf(err, err_n,
                "train.ppo_group_ids: not dense; group id %d is unused but %d "
                "is present", g, max_id);
            return 0;
        }
    }
    *num_groups = max_id + 1;
    if (*num_groups > PPO_MAX_GROUPS) {
        snprintf(err, err_n, "train.ppo_group_ids: %d groups exceeds "
            "PPO_MAX_GROUPS=%d", *num_groups, PPO_MAX_GROUPS);
        return 0;
    }
    return 1;
}

// Single startup validation entry point. Runs before any allocation.
//   mode_raw    train.ppo_clip_mode string
//   ids_raw     train.ppo_group_ids string ("-1" disables)
//   act_sizes   per-head class counts (1 means continuous in this codebase)
// On success writes mode, ids (num_atns entries, -1 when unused) and
// num_groups (0 for joint). Returns 1 on success, 0 with a message in err.
static inline int ppo_group_config_validate(const char* mode_raw,
        const char* ids_raw, int num_atns, const int* act_sizes,
        int is_continuous, int vtrace, int max_head_classes,
        int* mode_out, int* ids, int* num_groups, char* err, int err_n) {
    int mode = ppo_clip_mode_parse(mode_raw);
    if (mode < 0) {
        snprintf(err, err_n,
            "train.ppo_clip_mode: \"%s\" is not one of joint|per_head|"
            "group_sum|group_mean", mode_raw ? mode_raw : "(null)");
        return 0;
    }
    *mode_out = mode;
    *num_groups = 0;
    for (int i = 0; i < num_atns && i < PPO_MAX_HEADS; i++) {
        ids[i] = -1;
    }

    int disabled = ppo_group_ids_is_disabled(ids_raw);
    if (mode == PPO_CLIP_JOINT) {
        if (!disabled) {
            snprintf(err, err_n,
                "train.ppo_clip_mode=joint requires the disabled sentinel "
                "train.ppo_group_ids=-1 (got \"%s\")", ids_raw ? ids_raw : "(null)");
            return 0;
        }
        return 1;
    }

    // Grouped modes are categorical-only; joint keeps continuous support.
    if (is_continuous) {
        snprintf(err, err_n,
            "train.ppo_clip_mode=%s requires categorical actions; continuous "
            "and mixed action spaces are joint-only", ppo_clip_mode_name(mode));
        return 0;
    }
    for (int h = 0; h < num_atns; h++) {
        if (act_sizes[h] < 2) {
            snprintf(err, err_n,
                "train.ppo_clip_mode=%s: head %d has %d classes; grouped modes "
                "require categorical heads (>= 2 classes)",
                ppo_clip_mode_name(mode), h, act_sizes[h]);
            return 0;
        }
        if (act_sizes[h] > max_head_classes) {
            snprintf(err, err_n,
                "train.ppo_clip_mode=%s: head %d has %d classes > "
                "PPO_MAX_HEAD_A=%d", ppo_clip_mode_name(mode), h,
                act_sizes[h], max_head_classes);
            return 0;
        }
    }
    if (vtrace) {
        snprintf(err, err_n,
            "train.ppo_clip_mode=%s is incompatible with train.vtrace=1; "
            "V-trace importance is defined on the joint log-ratio",
            ppo_clip_mode_name(mode));
        return 0;
    }
    if (num_atns > PPO_MAX_HEADS) {
        snprintf(err, err_n,
            "train.ppo_clip_mode=%s: %d action heads exceeds PPO_MAX_HEADS=%d",
            ppo_clip_mode_name(mode), num_atns, PPO_MAX_HEADS);
        return 0;
    }

    if (mode == PPO_CLIP_PER_HEAD) {
        // per_head is one head per factor. The mapping is either disabled or
        // the explicit identity mapping; anything else is a contradiction.
        if (!disabled) {
            int tmp[PPO_MAX_HEADS];
            int tmp_groups = 0;
            if (!ppo_group_ids_parse(ids_raw, num_atns, tmp, &tmp_groups,
                    err, err_n)) {
                return 0;
            }
            for (int h = 0; h < num_atns; h++) {
                if (tmp[h] != h) {
                    snprintf(err, err_n,
                        "train.ppo_clip_mode=per_head requires "
                        "train.ppo_group_ids=-1 or the identity mapping "
                        "0,1,...,%d (head %d maps to group %d)",
                        num_atns - 1, h, tmp[h]);
                    return 0;
                }
            }
        }
        for (int h = 0; h < num_atns; h++) {
            ids[h] = h;
        }
        *num_groups = num_atns;
        return 1;
    }

    if (disabled) {
        snprintf(err, err_n,
            "train.ppo_clip_mode=%s requires an explicit dense 0-based "
            "train.ppo_group_ids mapping of length NUM_ATNS=%d",
            ppo_clip_mode_name(mode), num_atns);
        return 0;
    }
    return ppo_group_ids_parse(ids_raw, num_atns, ids, num_groups, err, err_n);
}

// Per-transition grouped statistics. All fields are already averaged over the
// row's active factors; the caller weights rows by 1/(N*T).
typedef struct {
    float pg_loss;
    float old_kl;
    float kl;
    float clipfrac;
    float importance;
    float abs_logratio;
    float logratio_max;
    float logratio_min;
    float ratio_max;
    int active_groups;
} PpoGroupStats;

// Forward + backward for one original env-agent transition.
//   logratio_sum[g]  sum over that group's ACTIVE members of (new_lp - old_lp)
//   active_count[g]  number of active members in group g (0 -> inactive group)
//   dlogp_out[g]     d(total loss)/d(new log-prob of any active member of g)
// Returns a PpoErr code; nonzero means the caller must fail closed.
PPO_HD int ppo_group_row(int mode, int num_groups,
        const float* logratio_sum, const int* active_count,
        float adv, float clip_coef, float inv_nt,
        float* dlogp_out, PpoGroupStats* stats) {
    stats->pg_loss = 0.0f;
    stats->old_kl = 0.0f;
    stats->kl = 0.0f;
    stats->clipfrac = 0.0f;
    stats->importance = 0.0f;
    stats->abs_logratio = 0.0f;
    stats->logratio_max = -INFINITY;
    stats->logratio_min = INFINITY;
    stats->ratio_max = -INFINITY;
    stats->active_groups = 0;

    int active = 0;
    for (int g = 0; g < num_groups; g++) {
        if (active_count[g] > 0) {
            active++;
        }
    }
    if (active == 0) {
        for (int g = 0; g < num_groups; g++) {
            dlogp_out[g] = 0.0f;
        }
        return PPO_ERR_NO_ACTIVE_FACTORS;
    }

    float inv_active = 1.0f / (float)active;
    float clip_lo = 1.0f - clip_coef;
    float clip_hi = 1.0f + clip_coef;
    for (int g = 0; g < num_groups; g++) {
        int cnt = active_count[g];
        if (cnt <= 0) {
            dlogp_out[g] = 0.0f;
            continue;
        }
        float lr = logratio_sum[g];
        if (mode == PPO_CLIP_GROUP_MEAN) {
            lr /= (float)cnt;
        }
        if (!isfinite(lr)) {
            return PPO_ERR_NONFINITE;
        }
        float ratio = expf(lr);
        if (!isfinite(ratio)) {
            return PPO_ERR_NONFINITE;
        }
        float ratio_clipped = fmaxf(clip_lo, fminf(clip_hi, ratio));
        float wa = -adv;
        float pg1 = wa * ratio;
        float pg2 = wa * ratio_clipped;
        float pg = fmaxf(pg1, pg2);
        float d_ratio = wa * inv_nt * inv_active;
        if (pg2 > pg1 && (ratio <= clip_lo || ratio >= clip_hi)) {
            d_ratio = 0.0f;
        }
        float d_logp = d_ratio * ratio;
        if (mode == PPO_CLIP_GROUP_MEAN) {
            d_logp /= (float)cnt;
        }
        dlogp_out[g] = d_logp;

        stats->pg_loss += pg;
        stats->old_kl += -lr;
        stats->kl += (ratio - 1.0f) - lr;
        stats->clipfrac += fabsf(ratio - 1.0f) > clip_coef ? 1.0f : 0.0f;
        stats->importance += ratio;
        stats->abs_logratio += fabsf(lr);
        stats->logratio_max = fmaxf(stats->logratio_max, lr);
        stats->logratio_min = fminf(stats->logratio_min, lr);
        stats->ratio_max = fmaxf(stats->ratio_max, ratio);
    }

    stats->pg_loss *= inv_active;
    stats->old_kl *= inv_active;
    stats->kl *= inv_active;
    stats->clipfrac *= inv_active;
    stats->importance *= inv_active;
    stats->abs_logratio *= inv_active;
    stats->active_groups = active;
    if (!isfinite(stats->pg_loss)) {
        return PPO_ERR_NONFINITE;
    }
    return PPO_ERR_NONE;
}

// Legal-action cardinality with fail-closed mask validation. Mask entries must
// be exactly 0 or 1 (existing env convention) and finite, and every head needs
// at least one legal action. Returns a PpoErr code; count lands in *out.
PPO_HD int ppo_mask_legal_count(const float* mask_row, int A, int* out) {
    int legal = 0;
    for (int a = 0; a < A; a++) {
        float m = mask_row[a];
        if (!isfinite(m) || (m != 0.0f && m != 1.0f)) {
            *out = 0;
            return PPO_ERR_MASK_NOT_BINARY;
        }
        legal += m != 0.0f ? 1 : 0;
    }
    *out = legal;
    if (legal == 0) {
        return PPO_ERR_MASK_NO_LEGAL;
    }
    return PPO_ERR_NONE;
}

// A head contributes policy/entropy gradient only when the env consumed it and
// it had a real choice. Deterministic (cardinality 1) heads are excluded.
PPO_HD int ppo_head_active(int consumed, int legal_count) {
    return consumed && legal_count > 1;
}

// Everything the grouped objective needs for one original env-agent
// transition. All mask/precision-dependent work is done by the caller, so the
// production kernel and the host tests run the exact same code below.
typedef struct {
    int num_atns;
    int num_groups;
    int mode;
    const int* act_sizes;       // [num_atns]
    const int* group_ids;       // [num_atns] head -> group
    const int* legal_count;     // [num_atns] legal-mask cardinality
    const int* consumed;        // [num_atns] env head-used semantics
    const float* actions;       // [num_atns] sampled indices
    const float* old_group_lp;  // [num_groups] rollout old group log-prob sums
    const float* head_lp_mix;   // [num_atns] or NULL: replacement new log-prob
    const float* head_grad_scale;  // [num_atns] or NULL: extra d_logp factor
    float adv;
    float clip_coef;
    float ent_coef;
    float inv_nt;
} PpoGroupRowInput;

// Forward + backward of the grouped surrogate for one transition.
// logps_inout holds per-action new log-probs on entry (A_total wide, heads
// concatenated) and receives the logit gradients on exit. Inactive heads —
// unused, consumed-but-inactive, and deterministic — are written to exactly
// zero. Entropy is averaged over active heads (no head-count weighting).
// CAP bounds the fixed-size group scratch; pass a compile-time head bound.
template <int CAP>
PPO_HD int ppo_group_row_apply(const PpoGroupRowInput* in, float* logps_inout,
        PpoGroupStats* stats, float* out_entropy, int* out_active_heads) {
    float g_logratio[CAP];
    float g_dlogp[CAP];
    int g_count[CAP];
    int nh = in->num_atns;
    int ng = in->num_groups;
    *out_entropy = 0.0f;
    *out_active_heads = 0;
    if (ng <= 0 || ng > CAP || nh <= 0) {
        return PPO_ERR_SHAPE;
    }
    for (int g = 0; g < ng; g++) {
        g_logratio[g] = 0.0f;
        g_dlogp[g] = 0.0f;
        g_count[g] = 0;
    }

    int err = PPO_ERR_NONE;
    int offset = 0;
    int n_active = 0;
    for (int h = 0; h < nh; h++) {
        int A = in->act_sizes[h];
        if (!ppo_head_active(in->consumed[h], in->legal_count[h])) {
            offset += A;
            continue;
        }
        int act = (int)in->actions[h];
        if (act < 0 || act >= A) {
            return PPO_ERR_ACTION_RANGE;
        }
        int gid = in->group_ids[h];
        if (gid < 0 || gid >= ng) {
            return PPO_ERR_SHAPE;
        }
        float lp = in->head_lp_mix ? in->head_lp_mix[h] : logps_inout[offset + act];
        g_logratio[gid] += lp;
        g_count[gid] += 1;
        n_active++;
        offset += A;
    }
    *out_active_heads = n_active;

    for (int g = 0; g < ng; g++) {
        if (g_count[g] > 0) {
            float old_lp = in->old_group_lp[g];
            if (!isfinite(old_lp)) {
                err = PPO_ERR_NONFINITE;
                old_lp = 0.0f;
            }
            g_logratio[g] -= old_lp;
        }
    }

    int code = ppo_group_row(in->mode, ng, g_logratio, g_count, in->adv,
        in->clip_coef, in->inv_nt, g_dlogp, stats);
    if (code != PPO_ERR_NONE) {
        err = code;
    }

    float inv_active = n_active > 0 ? 1.0f / (float)n_active : 0.0f;
    float d_ent = in->inv_nt * (-in->ent_coef) * inv_active;
    float total_entropy = 0.0f;
    offset = 0;
    for (int h = 0; h < nh; h++) {
        int A = in->act_sizes[h];
        if (!ppo_head_active(in->consumed[h], in->legal_count[h])) {
            for (int j = 0; j < A; j++) {
                logps_inout[offset + j] = 0.0f;
            }
            offset += A;
            continue;
        }
        int act = (int)in->actions[h];
        float ent = 0.0f;
        for (int j = 0; j < A; j++) {
            float logp = logps_inout[offset + j];
            ent -= expf(logp) * logp;
        }
        total_entropy += ent;
        float d_logp = g_dlogp[in->group_ids[h]];
        if (in->head_grad_scale) {
            d_logp *= in->head_grad_scale[h];
        }
        for (int j = 0; j < A; j++) {
            float logp = logps_inout[offset + j];
            float p = expf(logp);
            logps_inout[offset + j] = ((j == act ? 1.0f : 0.0f) - p) * d_logp
                + d_ent * p * (-ent - logp);
        }
        offset += A;
    }
    *out_entropy = total_entropy * inv_active;
    return err;
}
