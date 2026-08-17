"""Grouped PPO clipping: config validation + exact objective math.

These tests link against src/ppo_group.h through tests/ppo_group_shim.cc, i.e.
the same header the CUDA kernels in src/algo.cu compile, so the numbers proven
here are the numbers ppo_loss_compute produces. No GPU required.

Documented metric names emitted by the runtime (see src/pufferl.cu):
    loss/policy loss/value loss/entropy loss/total loss/old_kl loss/kl
    loss/clipfrac importance                      <- selected objective
    ppo/joint_old_kl ppo/joint_kl ppo/joint_clipfrac ppo/joint_importance
    ppo/joint_logratio ppo/active_groups ppo/active_heads
    ppo/logratio_absmean ppo/logratio_max ppo/logratio_min ppo/ratio_max
"""

import ctypes
import math
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
SRC = os.path.join(TESTS, "ppo_group_shim.cc")
HEADER = os.path.join(ROOT, "src", "ppo_group.h")
SO = os.path.join(TESTS, "ppo_group_shim.so")

JOINT, PER_HEAD, GROUP_SUM, GROUP_MEAN = 0, 1, 2, 3
ERR_NONE = 0
ERR_MASK_NOT_BINARY = 1
ERR_MASK_NO_LEGAL = 2
ERR_ACTION_RANGE = 3
ERR_NO_ACTIVE_FACTORS = 5
ERR_NONFINITE = 6
ERR_SHAPE = 7


def _build():
    cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        pytest.skip("no host C++ compiler available")
    if os.path.exists(SO):
        fresh = os.path.getmtime(SO)
        if fresh >= os.path.getmtime(SRC) and fresh >= os.path.getmtime(HEADER):
            return
    subprocess.check_call([cxx, "-std=c++17", "-O2", "-Wall", "-Wextra",
                           "-shared", "-fPIC", "-o", SO, SRC])


@pytest.fixture(scope="module")
def lib():
    _build()
    lib = ctypes.CDLL(SO)
    ci, cf, cp = ctypes.c_int, ctypes.c_float, ctypes.c_char_p
    ip, fp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
    lib.ppo_shim_clip_mode_parse.argtypes = [cp]
    lib.ppo_shim_group_ids_is_disabled.argtypes = [cp]
    lib.ppo_shim_config_validate.argtypes = [
        cp, cp, ci, ip, ci, ci, ci, ip, ip, ip, cp, ci]
    lib.ppo_shim_group_row.argtypes = [
        ci, ci, fp, ip, cf, cf, cf, fp, fp, ip]
    lib.ppo_shim_mask_legal_count.argtypes = [fp, ci, ip]
    lib.ppo_shim_head_active.argtypes = [ci, ci]
    lib.ppo_shim_max_heads.restype = ci
    lib.ppo_shim_max_groups.restype = ci
    lib.ppo_shim_group_row_apply.argtypes = [
        ci, ci, ci, ip, ip, ip, ip, fp, fp, cf, cf, cf, cf, fp, fp, fp, ip, ip]
    return lib


def _ints(values):
    return (ctypes.c_int * len(values))(*values)


def _floats(values):
    return (ctypes.c_float * len(values))(*values)


def validate(lib, mode, ids, num_atns=4, act_sizes=None, continuous=0,
             vtrace=0, max_head_classes=64):
    act_sizes = act_sizes if act_sizes is not None else [3] * num_atns
    mode_out = ctypes.c_int(-1)
    groups_out = ctypes.c_int(-1)
    out_ids = _ints([0] * max(num_atns, 1))
    err = ctypes.create_string_buffer(512)
    ok = lib.ppo_shim_config_validate(
        mode.encode(), ids.encode(), num_atns, _ints(act_sizes), continuous,
        vtrace, max_head_classes, ctypes.byref(mode_out), out_ids,
        ctypes.byref(groups_out), err, len(err))
    return ok, mode_out.value, list(out_ids)[:num_atns], groups_out.value, \
        err.value.decode()


def apply_row(lib, mode, act_sizes, group_ids, legal_count, consumed, actions,
              old_group_lp, logps, adv, clip_coef, ent_coef, inv_nt):
    num_atns = len(act_sizes)
    num_groups = len(old_group_lp)
    buf = _floats(logps)
    stats = _floats([0.0] * 9)
    entropy = ctypes.c_float(0.0)
    active_heads = ctypes.c_int(-1)
    active_groups = ctypes.c_int(-1)
    code = lib.ppo_shim_group_row_apply(
        mode, num_atns, num_groups, _ints(act_sizes), _ints(group_ids),
        _ints(legal_count), _ints(consumed), _floats(actions),
        _floats(old_group_lp), adv, clip_coef, ent_coef, inv_nt, buf, stats,
        ctypes.byref(entropy), ctypes.byref(active_heads),
        ctypes.byref(active_groups))
    return {
        "code": code,
        "grads": list(buf),
        "pg_loss": stats[0],
        "old_kl": stats[1],
        "kl": stats[2],
        "clipfrac": stats[3],
        "importance": stats[4],
        "abs_logratio": stats[5],
        "logratio_max": stats[6],
        "logratio_min": stats[7],
        "ratio_max": stats[8],
        "entropy": entropy.value,
        "active_heads": active_heads.value,
        "active_groups": active_groups.value,
    }


def log_softmax(logits):
    m = max(logits)
    lse = m + math.log(sum(math.exp(x - m) for x in logits))
    return [x - lse for x in logits]


# --------------------------------------------------------------------------
# Config semantics
# --------------------------------------------------------------------------

def test_default_config_is_joint_with_disabled_sentinel(lib):
    """Shipped defaults must keep the legacy joint path with no grouping."""
    text = open(os.path.join(ROOT, "config", "default.ini")).read()
    assert "ppo_clip_mode = joint" in text
    assert "ppo_group_ids = -1" in text

    ok, mode, ids, groups, err = validate(lib, "joint", "-1")
    assert ok == 1, err
    assert mode == JOINT
    assert groups == 0, "joint must not allocate grouped buffers"
    assert ids == [-1, -1, -1, -1]
    runtime = open(os.path.join(ROOT, "src", "pufferl.cu")).read()
    assert "if (hypers.ppo_num_groups > 0)" in runtime


def test_joint_rejects_an_explicit_mapping(lib):
    ok, _, _, _, err = validate(lib, "joint", "0,0,1,1")
    assert ok == 0
    assert "disabled sentinel" in err


@pytest.mark.parametrize("mode,expected", [
    ("joint", JOINT), ("group_sum", GROUP_SUM),
    ("per_head", PER_HEAD), ("group_mean", GROUP_MEAN),
])
def test_clip_mode_parse(lib, mode, expected):
    assert lib.ppo_shim_clip_mode_parse(mode.encode()) == expected


def test_cuda_shape_bounds_are_independent_and_exact(lib):
    assert lib.ppo_shim_max_heads() == 256
    assert lib.ppo_shim_max_groups() == 64


@pytest.mark.parametrize("bad", ["", "Joint", "group", "grouped_sum", "0", "-1"])
def test_invalid_clip_mode_rejected(lib, bad):
    assert lib.ppo_shim_clip_mode_parse(bad.encode()) == -1
    ok, _, _, _, err = validate(lib, bad, "-1")
    assert ok == 0
    assert "ppo_clip_mode" in err


def test_group_sum_accepts_non_contiguous_dense_mapping(lib):
    """Heads 0 and 2 share a group; ids need not be sorted or contiguous."""
    ok, mode, ids, groups, err = validate(lib, "group_sum", "1,0,1,2")
    assert ok == 1, err
    assert mode == GROUP_SUM
    assert ids == [1, 0, 1, 2]
    assert groups == 3


def test_group_sum_single_group(lib):
    ok, _, ids, groups, err = validate(lib, "group_mean", "0,0,0,0")
    assert ok == 1, err
    assert ids == [0, 0, 0, 0] and groups == 1


@pytest.mark.parametrize("ids,reason", [
    ("-1", "explicit dense"),          # sentinel not allowed for group modes
    ("0,2,1,4", "not dense"),          # hole at 3
    ("0,0,1", "expected NUM_ATNS"),    # too short
    ("0,0,1,1,2", "too many"),         # too long
    ("0,0,1,1.5", "nonnegative"),      # not an integer
    ("0,0,1,-1", "nonnegative"),       # negative id
    ("0,0,1,", "nonnegative"),         # trailing comma
    ("a,b,c,d", "nonnegative"),        # garbage
    ("1,1,1,1", "not dense"),          # 0 unused
])
def test_invalid_group_mappings_rejected(lib, ids, reason):
    ok, _, _, _, err = validate(lib, "group_sum", ids)
    assert ok == 0, f"{ids} should be rejected"
    assert reason in err, err


def test_group_id_at_or_above_the_cuda_bound_is_rejected(lib):
    cap = lib.ppo_shim_max_groups()
    ids = ",".join(["0", "0", "1", str(cap)])
    ok, _, _, _, err = validate(lib, "group_sum", ids)
    assert ok == 0
    assert "PPO_MAX_GROUPS" in err, err
    ok, _, _, _, err = validate(lib, "group_sum", "0,0,1,999999999999")
    assert ok == 0
    assert "PPO_MAX_GROUPS" in err, err


def test_per_head_accepts_sentinel_and_identity(lib):
    ok, mode, ids, groups, err = validate(lib, "per_head", "-1")
    assert ok == 1, err
    assert mode == PER_HEAD and ids == [0, 1, 2, 3] and groups == 4

    ok, _, ids, groups, err = validate(lib, "per_head", "0,1,2,3")
    assert ok == 1, err
    assert ids == [0, 1, 2, 3] and groups == 4


def test_per_head_rejects_non_identity_mapping(lib):
    ok, _, _, _, err = validate(lib, "per_head", "0,0,1,2")
    assert ok == 0
    assert "identity mapping" in err


@pytest.mark.parametrize("mode", ["group_sum", "per_head", "group_mean"])
def test_grouped_rejects_continuous(lib, mode):
    ids = "-1" if mode == "per_head" else "0,0,1,2"
    ok, _, _, _, err = validate(lib, mode, ids, continuous=1)
    assert ok == 0
    assert "categorical" in err


@pytest.mark.parametrize("mode", ["group_sum", "per_head", "group_mean"])
def test_grouped_rejects_vtrace(lib, mode):
    ids = "-1" if mode == "per_head" else "0,0,1,2"
    ok, _, _, _, err = validate(lib, mode, ids, vtrace=1)
    assert ok == 0
    assert "vtrace" in err


def test_grouped_rejects_single_class_head(lib):
    ok, _, _, _, err = validate(lib, "group_sum", "0,0,1,2",
                                act_sizes=[3, 1, 4, 2])
    assert ok == 0
    assert "categorical heads" in err


def test_grouped_rejects_head_wider_than_cuda_bound(lib):
    ok, _, _, _, err = validate(lib, "group_sum", "0,0,1,2",
                                act_sizes=[3, 2, 99, 2], max_head_classes=64)
    assert ok == 0
    assert "PPO_MAX_HEAD_A" in err


def test_grouped_rejects_too_many_heads(lib):
    n = lib.ppo_shim_max_heads() + 1
    ok, _, _, _, err = validate(lib, "per_head", "-1", num_atns=n,
                                act_sizes=[2] * n)
    assert ok == 0
    assert "PPO_MAX_HEADS" in err


def _kaggriculture_group_ids():
    # Six heads per group, permuted so adjacent heads are not grouped together.
    return [(h * 5) % 42 for h in range(252)]


def test_kaggriculture_per_head_accepts_252_heads(lib):
    n = 252
    ok, mode, ids, groups, err = validate(
        lib, "per_head", "-1", num_atns=n, act_sizes=[2] * n)
    assert ok == 1, err
    assert mode == PER_HEAD
    assert ids == list(range(n))
    assert groups == n

    identity = ",".join(str(h) for h in range(n))
    ok, _, ids, groups, err = validate(
        lib, "per_head", identity, num_atns=n, act_sizes=[2] * n)
    assert ok == 1, err
    assert ids == list(range(n)) and groups == n


@pytest.mark.parametrize("mode,expected", [
    ("group_sum", GROUP_SUM), ("group_mean", GROUP_MEAN),
])
def test_kaggriculture_group_modes_accept_252_heads_42_groups(
        lib, mode, expected):
    mapping = _kaggriculture_group_ids()
    ok, parsed, ids, groups, err = validate(
        lib, mode, ",".join(map(str, mapping)),
        num_atns=252, act_sizes=[2] * 252)
    assert ok == 1, err
    assert parsed == expected
    assert ids == mapping
    assert groups == 42
    assert all(mapping.count(g) == 6 for g in range(42))


def test_kaggriculture_rejects_non_dense_and_over_bound_mappings(lib):
    mapping = _kaggriculture_group_ids()
    non_dense = [18 if group == 17 else group for group in mapping]
    ok, _, _, _, err = validate(
        lib, "group_sum", ",".join(map(str, non_dense)),
        num_atns=252, act_sizes=[2] * 252)
    assert ok == 0
    assert "not dense" in err

    over_bound = list(mapping)
    over_bound[0] = lib.ppo_shim_max_groups()
    ok, _, _, _, err = validate(
        lib, "group_mean", ",".join(map(str, over_bound)),
        num_atns=252, act_sizes=[2] * 252)
    assert ok == 0
    assert "PPO_MAX_GROUPS" in err


@pytest.mark.parametrize("mode", [PER_HEAD, GROUP_SUM, GROUP_MEAN])
def test_kaggriculture_exact_shape_max_active_factors(lib, mode):
    n = 252
    mapping = list(range(n)) if mode == PER_HEAD \
        else _kaggriculture_group_ids()
    groups = n if mode == PER_HEAD else 42
    logps = [math.log(0.5)] * (2 * n)
    old_group_lp = [math.log(0.5)] * groups if mode == PER_HEAD \
        else [6 * math.log(0.5)] * groups
    got = apply_row(
        lib, mode, [2] * n, mapping, [2] * n, [1] * n, [0.0] * n,
        old_group_lp, logps, 1.0, 0.2, 0.0, 1.0)
    assert got["code"] == ERR_NONE
    assert got["active_heads"] == n
    assert got["active_groups"] == groups
    assert got["pg_loss"] == pytest.approx(-1.0, rel=1e-6)


def test_joint_allows_continuous_and_vtrace(lib):
    ok, mode, _, groups, err = validate(lib, "joint", "-1", continuous=1,
                                        vtrace=1, act_sizes=[1, 1, 1, 1])
    assert ok == 1, err
    assert mode == JOINT and groups == 0


# --------------------------------------------------------------------------
# Mask / activity semantics
# --------------------------------------------------------------------------

def test_mask_validation_and_cardinality(lib):
    out = ctypes.c_int(-1)
    assert lib.ppo_shim_mask_legal_count(_floats([1, 0, 1, 1]), 4,
                                         ctypes.byref(out)) == ERR_NONE
    assert out.value == 3
    assert lib.ppo_shim_mask_legal_count(_floats([0, 0, 0]), 3,
                                         ctypes.byref(out)) == ERR_MASK_NO_LEGAL
    assert lib.ppo_shim_mask_legal_count(_floats([1, 0.5]), 2,
                                         ctypes.byref(out)) == ERR_MASK_NOT_BINARY
    assert lib.ppo_shim_mask_legal_count(_floats([1, float("nan")]), 2,
                                         ctypes.byref(out)) == ERR_MASK_NOT_BINARY


def test_head_active_requires_consumed_and_stochastic(lib):
    assert lib.ppo_shim_head_active(1, 2) == 1
    assert lib.ppo_shim_head_active(1, 1) == 0   # deterministic head
    assert lib.ppo_shim_head_active(0, 5) == 0   # not consumed by the env
    assert lib.ppo_shim_head_active(0, 1) == 0


# --------------------------------------------------------------------------
# Objective math
# --------------------------------------------------------------------------

def _clip_surrogate(adv, ratio, clip):
    return max(-adv * ratio, -adv * min(max(ratio, 1 - clip), 1 + clip))


def test_group_sum_hand_computed(lib):
    """Two active heads in one group; a third head is deterministic."""
    adv, clip, inv_nt = 2.0, 0.2, 0.25
    lr = [0.3, -0.1]                       # per-head log-ratios
    counts = [2]
    total = sum(lr)
    ratio = math.exp(total)
    expected_pg = _clip_surrogate(adv, ratio, clip)

    dlogp = _floats([0.0])
    stats = _floats([0.0] * 9)
    groups = ctypes.c_int(-1)
    code = lib.ppo_shim_group_row(GROUP_SUM, 1, _floats([total]), _ints(counts),
                                  adv, clip, inv_nt, dlogp, stats,
                                  ctypes.byref(groups))
    assert code == ERR_NONE
    assert groups.value == 1
    assert stats[0] == pytest.approx(expected_pg, rel=1e-6)
    assert stats[1] == pytest.approx(-total, rel=1e-6)
    assert stats[2] == pytest.approx((ratio - 1) - total, rel=1e-6)
    assert stats[4] == pytest.approx(ratio, rel=1e-6)
    # ratio = e^0.2 = 1.2214 > 1 + clip, and the clipped branch wins for adv>0
    assert stats[3] == pytest.approx(1.0)
    assert dlogp[0] == pytest.approx(0.0), "clipped-out group gets zero grad"


@pytest.mark.parametrize("adv", [1.25, -1.25])
def test_group_sum_single_group_matches_legacy_joint_surrogate(lib, adv):
    """Summing several head log-ratios into one group is legacy joint geometry."""
    per_head = [0.12, -0.07, 0.03]
    joint_logratio = sum(per_head)
    ratio = math.exp(joint_logratio)
    clip, inv_nt = 0.2, 0.125
    stats = _floats([0.0] * 9)
    dlogp = _floats([0.0])
    groups = ctypes.c_int(-1)

    code = lib.ppo_shim_group_row(
        GROUP_SUM, 1, _floats([joint_logratio]), _ints([len(per_head)]),
        adv, clip, inv_nt, dlogp, stats, ctypes.byref(groups))

    assert code == ERR_NONE
    assert groups.value == 1
    assert stats[0] == pytest.approx(_clip_surrogate(adv, ratio, clip), rel=1e-6)
    assert stats[1] == pytest.approx(-joint_logratio, rel=1e-6)
    assert stats[2] == pytest.approx(
        (ratio - 1.0) - joint_logratio, rel=1e-6, abs=1e-7)
    assert stats[3] == pytest.approx(float(abs(ratio - 1.0) > clip), rel=1e-6)
    assert stats[4] == pytest.approx(ratio, rel=1e-6)


def test_group_mean_divides_by_active_members(lib):
    adv, clip, inv_nt = 1.0, 0.5, 1.0
    total, cnt = 0.6, 3
    mean_lr = total / cnt
    ratio = math.exp(mean_lr)
    dlogp = _floats([0.0])
    stats = _floats([0.0] * 9)
    groups = ctypes.c_int(-1)
    code = lib.ppo_shim_group_row(GROUP_MEAN, 1, _floats([total]), _ints([cnt]),
                                  adv, clip, inv_nt, dlogp, stats,
                                  ctypes.byref(groups))
    assert code == ERR_NONE
    assert stats[0] == pytest.approx(_clip_surrogate(adv, ratio, clip), rel=1e-6)
    # unclipped (ratio = e^0.2 < 1.5): d/d(member logp) = -adv * ratio / cnt
    assert dlogp[0] == pytest.approx(-adv * ratio / cnt, rel=1e-6)


def test_per_head_averages_over_active_factors(lib):
    adv, clip, inv_nt = -1.5, 0.2, 0.5
    lr = [0.05, 0.0, -0.4]
    counts = [1, 0, 1]           # middle factor inactive
    stats = _floats([0.0] * 9)
    dlogp = _floats([0.0] * 3)
    groups = ctypes.c_int(-1)
    code = lib.ppo_shim_group_row(PER_HEAD, 3, _floats(lr), _ints(counts),
                                  adv, clip, inv_nt, dlogp, stats,
                                  ctypes.byref(groups))
    assert code == ERR_NONE
    assert groups.value == 2
    r0, r2 = math.exp(lr[0]), math.exp(lr[2])
    expected = 0.5 * (_clip_surrogate(adv, r0, clip)
                      + _clip_surrogate(adv, r2, clip))
    assert stats[0] == pytest.approx(expected, rel=1e-6)
    assert stats[4] == pytest.approx(0.5 * (r0 + r2), rel=1e-6)
    assert stats[3] == pytest.approx(0.5), "only head 2 is outside the clip band"
    assert dlogp[1] == 0.0
    # head 0 unclipped: -adv * inv_nt / active_groups * ratio
    assert dlogp[0] == pytest.approx(-adv * inv_nt * 0.5 * r0, rel=1e-6)
    # head 2: ratio 0.67 <= 1 - clip and adv < 0, so the clipped branch wins
    # and the gradient is suppressed
    assert dlogp[2] == 0.0


def test_no_active_factor_fails_closed(lib):
    stats = _floats([0.0] * 9)
    dlogp = _floats([0.0] * 2)
    groups = ctypes.c_int(-1)
    code = lib.ppo_shim_group_row(GROUP_SUM, 2, _floats([0.0, 0.0]),
                                  _ints([0, 0]), 1.0, 0.2, 1.0, dlogp, stats,
                                  ctypes.byref(groups))
    assert code == ERR_NO_ACTIVE_FACTORS
    assert list(dlogp) == [0.0, 0.0]


def test_non_finite_logratio_fails_closed(lib):
    stats = _floats([0.0] * 9)
    dlogp = _floats([0.0])
    groups = ctypes.c_int(-1)
    code = lib.ppo_shim_group_row(GROUP_SUM, 1, _floats([float("inf")]),
                                  _ints([1]), 1.0, 0.2, 1.0, dlogp, stats,
                                  ctypes.byref(groups))
    assert code == ERR_NONFINITE


def test_huge_logratio_overflow_fails_closed(lib):
    stats = _floats([0.0] * 9)
    dlogp = _floats([0.0])
    groups = ctypes.c_int(-1)
    code = lib.ppo_shim_group_row(GROUP_SUM, 1, _floats([200.0]), _ints([1]),
                                  1.0, 0.2, 1.0, dlogp, stats,
                                  ctypes.byref(groups))
    assert code == ERR_NONFINITE, "exp overflow must not be clamped into success"


# --------------------------------------------------------------------------
# Full transition: activity, grouping, gradients, entropy
# --------------------------------------------------------------------------

ACT_SIZES = [3, 2, 4, 2]
GROUP_IDS = [1, 0, 1, 2]        # non-contiguous: heads 0 and 2 share group 1
LOGITS = [[0.5, -0.2, 0.1], [0.3, 0.7], [1.0, 0.0, -0.5, 0.25], [0.1, -0.1]]
ACTIONS = [2.0, 1.0, 0.0, 1.0]


def _row_logps():
    out = []
    for head in LOGITS:
        out.extend(log_softmax(head))
    return out


def _reference(mode, legal, consumed, old_group_lp, adv, clip, ent_coef,
               inv_nt):
    """Independent Python reference for the whole transition."""
    logps = _row_logps()
    offsets, off = [], 0
    for size in ACT_SIZES:
        offsets.append(off)
        off += size

    active = [bool(consumed[h] and legal[h] > 1) for h in range(4)]
    num_groups = max(GROUP_IDS) + 1 if mode != PER_HEAD else 4
    ids = GROUP_IDS if mode != PER_HEAD else [0, 1, 2, 3]
    sums = [0.0] * num_groups
    counts = [0] * num_groups
    for h in range(4):
        if active[h]:
            sums[ids[h]] += logps[offsets[h] + int(ACTIONS[h])]
            counts[ids[h]] += 1
    for g in range(num_groups):
        if counts[g]:
            sums[g] -= old_group_lp[g]

    n_active_groups = sum(1 for c in counts if c > 0)
    n_active_heads = sum(active)
    pg, dlogp = 0.0, [0.0] * num_groups
    for g in range(num_groups):
        if not counts[g]:
            continue
        lr = sums[g] / counts[g] if mode == GROUP_MEAN else sums[g]
        ratio = math.exp(lr)
        pg += _clip_surrogate(adv, ratio, clip)
        clipped = (-adv * min(max(ratio, 1 - clip), 1 + clip) > -adv * ratio
                   and (ratio <= 1 - clip or ratio >= 1 + clip))
        d = 0.0 if clipped else -adv * inv_nt / n_active_groups
        d *= ratio
        if mode == GROUP_MEAN:
            d /= counts[g]
        dlogp[g] = d
    pg /= n_active_groups

    d_ent = inv_nt * (-ent_coef) / n_active_heads
    grads = [0.0] * len(logps)
    entropy = 0.0
    for h in range(4):
        if not active[h]:
            continue
        base, size = offsets[h], ACT_SIZES[h]
        ent = -sum(math.exp(logps[base + j]) * logps[base + j]
                   for j in range(size))
        entropy += ent
        act = int(ACTIONS[h])
        for j in range(size):
            p = math.exp(logps[base + j])
            grads[base + j] = ((1.0 if j == act else 0.0) - p) * dlogp[ids[h]] \
                + d_ent * p * (-ent - logps[base + j])
    return {
        "pg_loss": pg,
        "grads": grads,
        "entropy": entropy / n_active_heads,
        "active_heads": n_active_heads,
        "active_groups": n_active_groups,
    }


@pytest.mark.parametrize("mode", [GROUP_SUM, PER_HEAD, GROUP_MEAN])
def test_full_row_matches_independent_reference(lib, mode):
    legal = [3, 1, 4, 2]         # head 1 is deterministic -> inactive
    consumed = [1, 1, 1, 0]      # head 3 not consumed by the env -> inactive
    adv, clip, ent_coef, inv_nt = 0.75, 0.2, 0.01, 1.0 / 8.0
    ids = GROUP_IDS if mode != PER_HEAD else [0, 1, 2, 3]
    num_groups = 3 if mode != PER_HEAD else 4
    old = [-0.9, -1.7, -0.4, -0.6][:num_groups]

    got = apply_row(lib, mode, ACT_SIZES, ids, legal, consumed, ACTIONS, old,
                    _row_logps(), adv, clip, ent_coef, inv_nt)
    ref = _reference(mode, legal, consumed, old, adv, clip, ent_coef, inv_nt)

    assert got["code"] == ERR_NONE
    assert got["active_heads"] == ref["active_heads"] == 2
    assert got["active_groups"] == ref["active_groups"]
    assert got["pg_loss"] == pytest.approx(ref["pg_loss"], rel=1e-5)
    assert got["entropy"] == pytest.approx(ref["entropy"], rel=1e-5)
    for i, (a, b) in enumerate(zip(got["grads"], ref["grads"])):
        assert a == pytest.approx(b, rel=1e-4, abs=1e-7), f"grad {i}"


def test_deterministic_and_unconsumed_heads_get_zero_gradient(lib):
    legal = [3, 1, 4, 2]
    consumed = [1, 1, 1, 0]
    got = apply_row(lib, GROUP_SUM, ACT_SIZES, GROUP_IDS, legal, consumed,
                    ACTIONS, [-0.9, -1.7, -0.4], _row_logps(), 0.75, 0.2, 0.01,
                    0.125)
    grads = got["grads"]
    # head 1 (deterministic, cardinality 1) and head 3 (not consumed)
    assert grads[3:5] == [0.0, 0.0]
    assert grads[9:11] == [0.0, 0.0]
    # active heads 0 and 2 do receive gradient
    assert any(abs(v) > 0 for v in grads[0:3])
    assert any(abs(v) > 0 for v in grads[5:9])


def test_inactive_consumed_head_excluded_from_group_sum(lib):
    """A consumed head with cardinality 1 must not enter its group's ratio."""
    logps = _row_logps()
    old = [-0.9, -1.7, -0.4]
    args = dict(act_sizes=ACT_SIZES, group_ids=GROUP_IDS, actions=ACTIONS,
                old_group_lp=old, adv=1.0, clip_coef=0.2, ent_coef=0.0,
                inv_nt=1.0)
    # head 2 deterministic -> group 1 carries head 0 only
    got = apply_row(lib, GROUP_SUM, legal_count=[3, 2, 1, 2],
                    consumed=[1, 1, 1, 1], logps=list(logps), **args)
    lr_head0 = logps[0 + int(ACTIONS[0])]
    # groups 0 (head 1) and 1 (head 0) are active; head 3 -> group 2 active too
    assert got["active_groups"] == 3
    assert got["active_heads"] == 3
    ratio_g1 = math.exp(lr_head0 - old[1])
    # importance is the mean over active groups; recover group 1's contribution
    r_g0 = math.exp(logps[3 + int(ACTIONS[1])] - old[0])
    r_g2 = math.exp(logps[9 + int(ACTIONS[3])] - old[2])
    assert got["importance"] == pytest.approx((r_g0 + ratio_g1 + r_g2) / 3.0,
                                              rel=1e-5)


def test_group_sum_and_per_head_agree_on_singleton_groups(lib):
    """With one head per group the two objectives are identical by definition."""
    legal, consumed = [3, 2, 4, 2], [1, 1, 1, 1]
    ids = [0, 1, 2, 3]
    old = [-0.9, -0.7, -1.4, -0.6]
    kwargs = dict(act_sizes=ACT_SIZES, group_ids=ids, legal_count=legal,
                  consumed=consumed, actions=ACTIONS, old_group_lp=old,
                  adv=0.4, clip_coef=0.2, ent_coef=0.02, inv_nt=0.25)
    a = apply_row(lib, GROUP_SUM, logps=_row_logps(), **kwargs)
    b = apply_row(lib, PER_HEAD, logps=_row_logps(), **kwargs)
    c = apply_row(lib, GROUP_MEAN, logps=_row_logps(), **kwargs)
    assert a["pg_loss"] == pytest.approx(b["pg_loss"], rel=1e-6)
    assert a["pg_loss"] == pytest.approx(c["pg_loss"], rel=1e-6)
    assert a["grads"] == pytest.approx(b["grads"], rel=1e-6, abs=1e-9)
    assert a["grads"] == pytest.approx(c["grads"], rel=1e-6, abs=1e-9)


def test_single_active_factor_reproduces_joint_geometry(lib):
    """One active head, one group: exactly the legacy clipped joint surrogate."""
    legal, consumed = [3, 1, 1, 1], [1, 1, 1, 1]
    old = [0.0, -1.1, 0.0]
    adv, clip, inv_nt = 1.25, 0.2, 0.5
    got = apply_row(lib, GROUP_SUM, ACT_SIZES, GROUP_IDS, legal, consumed,
                    ACTIONS, old, _row_logps(), adv, clip, 0.0, inv_nt)
    logps = _row_logps()
    logratio = logps[int(ACTIONS[0])] - old[1]
    ratio = math.exp(logratio)
    assert got["active_groups"] == 1 and got["active_heads"] == 1
    assert got["pg_loss"] == pytest.approx(_clip_surrogate(adv, ratio, clip),
                                           rel=1e-6)
    assert got["kl"] == pytest.approx((ratio - 1) - logratio, rel=1e-5)
    assert got["old_kl"] == pytest.approx(-logratio, rel=1e-5)
    assert got["logratio_max"] == pytest.approx(logratio, rel=1e-6)
    assert got["logratio_min"] == pytest.approx(logratio, rel=1e-6)
    assert got["ratio_max"] == pytest.approx(ratio, rel=1e-6)


def test_non_finite_old_group_logprob_fails_closed(lib):
    got = apply_row(lib, GROUP_SUM, ACT_SIZES, GROUP_IDS, [3, 2, 4, 2],
                    [1, 1, 1, 1], ACTIONS, [-0.9, float("nan"), -0.4],
                    _row_logps(), 1.0, 0.2, 0.0, 1.0)
    assert got["code"] == ERR_NONFINITE


def test_out_of_range_action_fails_closed(lib):
    got = apply_row(lib, GROUP_SUM, ACT_SIZES, GROUP_IDS, [3, 2, 4, 2],
                    [1, 1, 1, 1], [7.0, 1.0, 0.0, 1.0], [-0.9, -1.7, -0.4],
                    _row_logps(), 1.0, 0.2, 0.0, 1.0)
    assert got["code"] == ERR_ACTION_RANGE


def test_zero_groups_is_a_shape_error(lib):
    """Joint mode never calls the grouped path; a zero-group call must fail."""
    got = apply_row(lib, GROUP_SUM, ACT_SIZES, GROUP_IDS, [3, 2, 4, 2],
                    [1, 1, 1, 1], ACTIONS, [], _row_logps(), 1.0, 0.2, 0.0, 1.0)
    assert got["code"] == ERR_SHAPE


# --------------------------------------------------------------------------
# Metric-name contract
# --------------------------------------------------------------------------

DOCUMENTED_METRICS = [
    "ppo/joint_old_kl", "ppo/joint_kl", "ppo/joint_clipfrac",
    "ppo/joint_importance", "ppo/joint_logratio", "ppo/active_groups",
    "ppo/active_heads", "ppo/logratio_absmean", "ppo/logratio_max",
    "ppo/logratio_min", "ppo/ratio_max", "ppo/clip_mode",
    "loss/policy", "loss/value", "loss/entropy", "loss/total", "loss/old_kl",
    "loss/kl", "loss/clipfrac", "importance",
]


def test_stable_capability_marker():
    text = open(HEADER).read()
    assert '#define PUFFER_GROUPED_PPO_CAPABILITY "pufferlib.grouped_ppo.v1"' in text
    assert "#define PUFFER_GROUPED_PPO_CAPABILITY_VERSION 1" in text


def test_documented_metric_names_are_emitted():
    text = open(os.path.join(ROOT, "src", "pufferl.cu")).read()
    for name in DOCUMENTED_METRICS:
        assert f'"{name}"' in text, f"missing metric {name}"
