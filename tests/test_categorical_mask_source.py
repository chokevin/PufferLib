"""Host-side structural checks for the exact action-mask runtime.

These run anywhere (no CUDA needed) and pin down the ordering guarantees that
cannot be expressed in a unit test of the primitive itself:

* src/categorical.cuh is included after the precision typedefs / ENV_HEADER and
  before the textual include of src/algo.cu;
* no rollout row reaches puf_step, and no learner row reaches GAE / the loss
  reduction / backward / the optimizer, before the categorical status check;
* the finite-sentinel (-1e4) masking and the fixed PPO_MAX_HEAD_A cache are gone
  from the categorical probability / entropy / gradient code;
* rollout sampling and learner recomputation both go through the shared
  primitive with warp-per-row launch geometry;
* the legacy ENV_SAMPLE_LOGITS call ABI and launch geometry are preserved and
  its precision logprobs are converted into the float32 PPO shadow;
* CUDA graph capture is split/disabled exactly where it would otherwise cross a
  fail-before-consume boundary.

Run: pytest -q tests/test_categorical_mask_source.py
"""
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")


def read(name):
    with open(os.path.join(SRC, name), "r") as handle:
        return handle.read()


PUFFERL = read("pufferl.cu")
ALGO = read("algo.cu")
CATEGORICAL = read("categorical.cuh")


def strip_comments(text):
    """Drop // and /* */ comments so structural checks ignore prose."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


CATEGORICAL_CODE = strip_comments(CATEGORICAL)
PUFFERL_CODE = strip_comments(PUFFERL)
ALGO_CODE = strip_comments(ALGO)


def body(text, signature, end_marker=None):
    """Return the source of the function starting at `signature`."""
    start = text.index(signature)
    depth = 0
    i = text.index("{", start)
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    raise AssertionError(f"unterminated body for {signature}")


def order(text, *needles):
    """Assert the needles appear in the given order, each exactly findable."""
    last = -1
    for needle in needles:
        idx = text.find(needle, last + 1)
        assert idx > last, f"expected {needle!r} after position {last}"
        last = idx
    return True


# --------------------------------------------------------------------------
# Compile order
# --------------------------------------------------------------------------
def test_primitive_included_after_env_header_and_before_algo():
    assert order(PUFFERL,
                 "typedef __nv_bfloat16 precision_t;",
                 "#include ENV_HEADER",
                 '#include "categorical.cuh"',
                 '#include "algo.cu"')


def test_primitive_is_environment_independent():
    for token in ("NUM_ATNS", "ACT_SIZES", "PUFFER_NETHACK",
                  "KG_HEAD_GATING_SELECTOR_VALUES", "nethack_", "kg_puffer5"):
        assert token not in CATEGORICAL_CODE, f"{token} leaked into the primitive"


def test_primitive_has_no_block_barriers_or_width_arrays():
    assert "__syncthreads" not in CATEGORICAL_CODE
    # House style (SKILL_ISSUES.md): CUDA sources avoid templates and the STL.
    assert "template" not in CATEGORICAL_CODE
    assert "std::" not in CATEGORICAL_CODE
    # No fixed-size thread-local arrays sized by head width or head count.
    assert not re.search(r"float\s+\w+\[[A-Za-z_0-9]+\]", CATEGORICAL_CODE)
    assert "PUF_CAT_WARP 32" in CATEGORICAL_CODE


# --------------------------------------------------------------------------
# Exactness / no finite sentinel
# --------------------------------------------------------------------------
def test_finite_sentinel_masking_is_gone():
    for name, text in (("algo.cu", ALGO_CODE), ("pufferl.cu", PUFFERL_CODE),
                       ("categorical.cuh", CATEGORICAL_CODE)):
        assert "-1e4f" not in text, f"finite mask sentinel still in {name}"
        assert "load_logit_masked" not in text
        assert "ppo_discrete_logsumexp" not in text
        assert "PPO_MAX_HEAD_A" not in text


def test_masked_categories_get_literal_zero():
    write_logps = body(CATEGORICAL, "__device__ void puf_cat_write_logps")
    assert "puf_cat_logp(to_float(logits[a]), n) : 0.0f;" in write_logps
    grads = body(CATEGORICAL, "__device__ void puf_cat_write_grads")
    assert "if (!puf_cat_legal(mask[a])) {\n            grad[a] = 0.0f;" in grads


def test_sampler_and_learner_share_one_normalization():
    sampler = body(PUFFERL, "__global__ void sample_logits(")
    learner = body(ALGO, "__global__ void cache_imp_and_v(")
    for text in (sampler, learner):
        assert "puf_cat_head_norm(" in text
    assert "puf_cat_logp_at(" in sampler
    assert "puf_cat_logp_at(" in learner
    # Both are launched one warp per row.
    assert "sample_logits<<<grid_size(n * PUF_CAT_WARP)" in PUFFERL
    assert "cache_imp_and_v<<<grid_size(B * TT * PUF_CAT_WARP)" in ALGO
    assert "ppo_categorical_grad<<<grad_grid" in ALGO


def test_float32_old_logprob_shadow_exists():
    assert "Float logprobs_f32;" in PUFFERL
    assert "alloc_register(alloc, &bufs->logprobs_f32);" in PUFFERL
    assert "view.logprobs_f32 = puf_time_view(base->logprobs_f32" in PUFFERL
    assert "rollouts->logprobs_f32.data, src.logprobs_f32.data" in PUFFERL
    assert "Float mb_logprobs_f32;" in ALGO
    assert ".old_logprobs = graph.mb_logprobs_f32.data," in ALGO
    # Legacy precision logprob buffer is retained for the custom sampler ABI.
    assert "Prec logprobs;" in PUFFERL
    assert "Prec mb_logprobs;" in ALGO


# --------------------------------------------------------------------------
# Fail before consume
# --------------------------------------------------------------------------
def test_rollout_status_checked_before_cpu_env_step():
    worker = body(PUFFERL, "static void* vec_thread_main(void* arg)")
    assert order(worker,
                 "pufferl_forward(pufferl, buf, t, stream)",
                 "cudaStreamSynchronize(stream)",
                 "pufferl_rollout_status_check(pufferl, buf)",
                 "puf_step(&envs[i])")


def test_rollout_status_checked_before_gpu_env_step():
    start = body(PUFFERL, "static void rollout_start(PuffeRL* p)")
    discrete = start[:start.index("bool first = p->hypers.cudagraphs")]
    assert order(discrete,
                 "if (!p->is_continuous || PUF_CUSTOM_SAMPLE_LOGITS)",
                 "pufferl_forward(p, 0, t, stream)",
                 "cudaStreamSynchronize(stream)",
                 "pufferl_rollout_status_check(p, 0)",
                 "puf_step(p->vec->envs)")
    # The captured whole-horizon graph is only used for continuous actions,
    # where the categorical path is never taken.
    assert "cudaStreamBeginCapture" not in discrete


def test_learner_status_checked_before_gae_loss_backward_and_optimizer():
    epoch = body(PUFFERL, "static void train_epoch_gpu(PuffeRL* pufferl")
    assert order(epoch,
                 "puf_cat_status_reset(pufferl->cat_status_train.device, stream)",
                 "arch_forward_train(",
                 "puf_cat_status_fetch(pufferl->cat_status_train.host",
                 "cudaStreamSynchronize(stream)",
                 'puf_cat_status_check(pufferl->cat_status_train.host',
                 "puff_advantage<<<",
                 "ppo_loss_fwd_bwd(",
                 "arch_backward(",
                 "muon_step(")


def test_status_reset_before_each_launch():
    forward = body(PUFFERL, "static void pufferl_forward_step(PuffeRL* pufferl")
    assert order(forward,
                 "puf_cat_status_reset(status, stream)",
                 "sample_logits<<<")
    # The status copy is issued outside the captured region.
    launch = body(PUFFERL, "void pufferl_forward(PuffeRL* pufferl, int buf")
    assert order(launch,
                 "cudaStreamEndCapture",
                 "pufferl_rollout_status_fetch(pufferl, buf, stream)")


def test_train_graphs_disabled_for_discrete_actions():
    impl = body(PUFFERL, "void train_impl(PuffeRL* pufferl, RolloutBuf* src_arg)")
    normalized = re.sub(r"\s+", " ", impl)
    assert ("bool train_graphs = hypers->cudagraphs && pufferl->is_continuous "
            "&& !PUF_CUSTOM_SAMPLE_LOGITS;") in normalized
    assert "bool first = train_graphs && pufferl->train_cudagraph[slot] == NULL;" in impl
    assert "if (train_graphs && !first)" in impl


def test_error_fallbacks_are_finite_and_in_range():
    create = body(PUFFERL, "PuffeRL* create_pufferl(Ini* ini, TrainContext* ctx)")
    assert order(create,
                 "if (act_sizes[i] <= 0)",
                 '"[puffer] fatal action-space error: head %d has width %d\\n"',
                 "abort();")

    sampler = body(PUFFERL, "__global__ void sample_logits(")
    fallback = sampler[sampler.index("if (err_code != PUF_CAT_OK)"):]
    assert "actions[idx * num_atns + h] = 0.0f;" in fallback
    assert "env_actions[idx * num_atns + h] = 0.0f;" in fallback
    assert "logprobs_f32[idx] = 0.0f;" in fallback
    assert "isfinite(v) ? v : 0.0f" in fallback
    assert "-1" not in fallback.split("puf_cat_report")[0].replace("-1;", "")

    learner = body(ALGO, "__global__ void cache_imp_and_v(")
    lf = learner[learner.index("if (err_code != PUF_CAT_OK)"):]
    assert "logps[at_base + a] = 0.0f;" in lf
    assert "imp_out[idx] = from_float(1.0f);" in lf
    assert "entropy_out[idx] = 0.0f;" in lf
    assert "isfinite(v) ? v : 0.0f" in lf


def test_every_error_code_is_reported_somewhere():
    reported = set()
    for text in (PUFFERL, ALGO, CATEGORICAL):
        for code in re.findall(r"PUF_CAT_ERR_[A-Z_]+", text):
            reported.add(code)
    for code in ("PUF_CAT_ERR_WIDTH", "PUF_CAT_ERR_EMPTY",
                 "PUF_CAT_ERR_LEGAL_NONFINITE", "PUF_CAT_ERR_ACTION",
                 "PUF_CAT_ERR_ACTION_ILLEGAL", "PUF_CAT_ERR_OLD_LOGPROB"):
        assert code in reported
    check = body(CATEGORICAL, "void puf_cat_status_check(")
    assert "abort();" in check
    assert "row %d, head %d, detail %d" in check


# --------------------------------------------------------------------------
# Public API / legacy hook compatibility
# --------------------------------------------------------------------------
LEGACY_ARGS = [
    "dec, p_logstd, pufferl->act_sizes,",
    "act_b.data, env->actions.data + (long)sub * act_cols,",
    "lp_b.data, val_b.data,",
    "pufferl->rng_states[buf] + off,",
    "mask_b.data, mask_stride);",
]


def test_legacy_sample_hook_abi_and_geometry_preserved():
    forward = body(PUFFERL, "static void pufferl_forward_step(PuffeRL* pufferl")
    legacy = forward[forward.index("#ifdef ENV_SAMPLE_LOGITS"):
                     forward.index("#else", forward.index("#ifdef ENV_SAMPLE_LOGITS"))]
    assert "ENV_SAMPLE_LOGITS<<<n, BLOCK_SIZE, 0, stream>>>(" in legacy
    assert order(legacy, *LEGACY_ARGS)
    # Legacy precision logprobs are converted + validated into the shadow.
    assert "logprob_shadow_from_precision<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(" in legacy
    assert "lp32_b.data, lp_b.data, n, status);" in legacy
    assert "#ifdef ENV_ACTION_KERNEL_HEADER" in PUFFERL
    assert "#include ENV_ACTION_KERNEL_HEADER" in PUFFERL


def test_legacy_hook_fixture_matches_launch_arguments():
    with open(os.path.join(HERE, "legacy_sample_hook.cuh"), "r") as handle:
        hook = handle.read()
    signature = hook[hook.index("__global__ void legacy_sample_logits("):
                     hook.index(") {")]
    for param in ("Prec dec_out", "Prec logstd", "int* act_sizes",
                  "float* actions", "float* env_actions",
                  "precision_t* logprobs", "precision_t* value_out",
                  "curandStatePhilox4_32_10_t* rng_states",
                  "precision_t* action_mask", "int mask_stride"):
        assert param in signature, param


def test_mask_semantics_stay_zero_illegal_nonzero_legal():
    assert ("__device__ __forceinline__ bool puf_cat_legal(precision_t m) {\n"
            "    return to_float(m) != 0.0f;\n}") in CATEGORICAL
    # Env byte masks reach the primitive through the trainer's cast kernel.
    assert "cast<<<grid_size(block_size * mask_size), BLOCK_SIZE, 0, stream>>>(" in PUFFERL
    # Env-facing buffers keep their types.
    assert "Byte action_mask; // (total_agents, mask_size); always allocated" in PUFFERL
    assert "unsigned char* action_mask;" in PUFFERL


def test_primary_rollout_transpose_uses_64_bit_indexing():
    assert "int grid_size(int64_t N)" in PUFFERL
    for signature in (
        "__global__ void transpose_primary_102(precision_t* dst",
        "__global__ void transpose_primary_102(float* dst",
    ):
        transpose = body(PUFFERL, signature)
        assert "int64_t train_agents" in transpose
        assert "int64_t idx" in transpose
        assert "int64_t total = (int64_t)T * train_agents * C;" in transpose
        assert "int64_t src_agent" in transpose
    assert "(int64_t)T * train_agents * obs_size" in PUFFERL
    assert "(int64_t)T * train_agents * num_atns" in PUFFERL
    assert "(int64_t)T * train_agents * mask_c" in PUFFERL
    assert 128 * 8192 * 7477 > 2**31 - 1


def test_environment_specific_gating_stays_caller_side():
    for hook in ("nethack_head_used", "nethack_verb_eps_for_mask",
                 "kg_puffer5_head_used"):
        assert hook in PUFFERL_CODE or hook in ALGO_CODE
        assert hook not in CATEGORICAL_CODE
    # Nethack epsilon mixing still reaches the sampler as a generic parameter.
    sampler = body(PUFFERL, "__global__ void sample_logits(")
    assert order(sampler,
                 "eps = nethack_verb_eps_for_mask(",
                 "PufCatNorm norm = puf_cat_head_norm(",
                 "head_logits, head_mask, A, eps > 0.0f);")
    assert "puf_cat_sample(\n            head_logits, head_mask, A, norm, rand_val, eps, inv_K)" in sampler
    learner = body(ALGO, "__global__ void cache_imp_and_v(")
    assert order(learner,
                 "nethack_verb_eps_for_mask(head_mask, A, eps, &inv_K)",
                 "PufCatNorm norm = puf_cat_head_norm(",
                 "head_logits, head_mask, A, eps > 0.0f);")
    assert "KG_HEAD_GATING_SELECTOR_VALUES" in sampler


def test_nethack_mixture_math_and_epsilon_are_shared_with_rollout_slot():
    assert "PufCatUniformMix puf_cat_uniform_mix(" in CATEGORICAL
    assert "Float policy_eps;" in PUFFERL
    assert re.search(
        r"view\.policy_eps\s*=\s*puf_time_view\(base->policy_eps", PUFFERL)
    assert "rollouts->policy_eps.data, src.policy_eps.data" in PUFFERL
    assert "Float mb_policy_eps;" in ALGO
    assert "graph.mb_policy_eps = slice_rows(rollouts->policy_eps" in PUFFERL

    sampler = body(PUFFERL, "__global__ void sample_logits(")
    learner = body(ALGO, "__global__ void cache_imp_and_v(")
    gradient = body(ALGO, "__global__ void ppo_categorical_grad(")
    for text in (sampler, learner, gradient):
        assert "puf_cat_uniform_mix(" in text
    assert "nethack_verb_train_logp" not in ALGO_CODE
    assert "nethack_verb_train_logp" not in PUFFERL_CODE
    assert "policy_eps[idx]" in sampler
    assert "rollout_eps[idx]" in learner
    assert "rollout_eps[idx]" in gradient


def test_custom_sampler_status_checks_cover_continuous_graph_paths():
    assert "PUF_CUSTOM_SAMPLE_LOGITS" in PUFFERL_CODE
    rollout = body(PUFFERL, "static void rollout_start(PuffeRL* p)")
    assert "if (!p->is_continuous || PUF_CUSTOM_SAMPLE_LOGITS)" in rollout
    epoch = body(PUFFERL, "static void train_epoch_gpu(PuffeRL* pufferl")
    assert "if (!pufferl->is_continuous || PUF_CUSTOM_SAMPLE_LOGITS)" in epoch
    impl = body(PUFFERL, "void train_impl(PuffeRL* pufferl, RolloutBuf* src_arg)")
    normalized = re.sub(r"\s+", " ", impl)
    assert ("bool train_graphs = hypers->cudagraphs && pufferl->is_continuous "
            "&& !PUF_CUSTOM_SAMPLE_LOGITS;") in normalized
    create = body(PUFFERL, "PuffeRL* create_pufferl(Ini* ini, TrainContext* ctx)")
    create_normalized = re.sub(r"\s+", " ", create)
    assert ("if (hypers.cudagraphs && (PUF_BACKEND != PUF_GPU "
            "|| !is_continuous || PUF_CUSTOM_SAMPLE_LOGITS))") in create_normalized

    hook_path = os.path.join(HERE, "nonfinite_sample_hook.cuh")
    assert os.path.exists(hook_path)
    with open(hook_path, "r") as handle:
        hook = handle.read()
    assert "logprobs[idx] = from_float(NAN);" in hook


def test_continuous_path_preserved():
    sampler = body(PUFFERL, "__global__ void sample_logits(")
    assert "ppo_continuous_head(mean, log_std, stored, &lp, &ent);" in sampler
    assert "curand_normal(&state)" in sampler
    learner = body(ALGO, "__global__ void cache_imp_and_v(")
    assert "safe_continuous_logstd(logstd.data, h)," in learner
    loss = body(ALGO, "__global__ void ppo_loss_compute(")
    assert "a.grad_logstd[nt * a.num_atns + h] =" in loss
