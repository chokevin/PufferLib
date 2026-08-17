"""Adversarial coverage for the exact categorical action-mask primitive
(src/categorical.cuh) via tests/categorical_mask_shim.cu.

Every case is compared against a float64 legal-only reference. The shim is
built twice: float (PRECISION_FLOAT) and bfloat16, matching the two production
precisions. Requires nvcc + a CUDA device; skips cleanly otherwise.

Run: pytest -q tests/test_categorical_mask.py
"""
import ctypes
import glob
import os
import shutil
import subprocess

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHIM = os.path.join(HERE, "categorical_mask_shim.cu")
HEADER = os.path.join(ROOT, "src", "categorical.cuh")

NVCC = shutil.which("nvcc")
if NVCC is None:
    pytest.skip("nvcc not available", allow_module_level=True)

PUF_CAT_OK = 0
PUF_CAT_ERR_WIDTH = 1
PUF_CAT_ERR_EMPTY = 2
PUF_CAT_ERR_LEGAL_NONFINITE = 3
PUF_CAT_ERR_ACTION = 4
PUF_CAT_ERR_ACTION_ILLEGAL = 5
PUF_CAT_ERR_OLD_LOGPROB = 6


def _build(float_precision):
    name = "categorical_shim_float.so" if float_precision else "categorical_shim_bf16.so"
    out = os.path.join(HERE, name)
    if (os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(SHIM)
            and os.path.getmtime(out) >= os.path.getmtime(HEADER)):
        return out
    cmd = [
        NVCC, "-shared", "-o", out, SHIM, "-std=c++17", "-O2",
        "-arch=" + os.environ.get("NVCC_ARCH", "native"),
        "--fmad=false", "-Xcompiler=-ffp-contract=off",
        "-Xcompiler=-fPIC",
    ]
    if float_precision:
        cmd.append("-DPRECISION_FLOAT")
    subprocess.run(cmd, check=True)
    return out


def _load(float_precision):
    lib = ctypes.CDLL(_build(float_precision))
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.puf_cat_shim_uses_bf16.restype = ctypes.c_int
    lib.puf_cat_shim_device_ok.restype = ctypes.c_int
    lib.puf_cat_shim_cache.argtypes = [
        ctypes.c_int, ctypes.c_int, i32, f32, u8, f32, f32,
        f32, f32, f32, f32, i32,
    ]
    lib.puf_cat_shim_cache.restype = ctypes.c_int
    lib.puf_cat_shim_sample.argtypes = [
        ctypes.c_int, ctypes.c_int, i32, f32, u8, f32, ctypes.c_float,
        f32, f32, i32,
    ]
    lib.puf_cat_shim_sample.restype = ctypes.c_int
    lib.puf_cat_shim_grad.argtypes = [
        ctypes.c_int, ctypes.c_int, i32, f32, u8, f32, f32, ctypes.c_float, f32,
    ]
    lib.puf_cat_shim_grad.restype = ctypes.c_int
    return lib


@pytest.fixture(scope="module", params=[True, False], ids=["float", "bf16"])
def lib(request):
    try:
        handle = _load(request.param)
    except subprocess.CalledProcessError as exc:  # nvcc present, build failed
        pytest.fail(f"shim build failed: {exc}")
    if not handle.puf_cat_shim_device_ok():
        pytest.skip("no CUDA device available")
    return handle


def is_bf16(lib):
    return bool(lib.puf_cat_shim_uses_bf16())


def quantize(lib, logits):
    """Round host logits the way the device precision will."""
    if not is_bf16(lib):
        return logits.astype(np.float32)
    raw = logits.astype(np.float32).view(np.uint32)
    # round-to-nearest-even bf16, matching __float2bfloat16 for finite values
    rounded = ((raw + 0x7FFF + ((raw >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    out = rounded.view(np.float32).copy()
    nan = ~np.isfinite(logits.astype(np.float32))
    out[nan] = logits.astype(np.float32)[nan]
    return out


def ref_head(logits, mask):
    """float64 legal-only reference: per-category logp and entropy."""
    legal = np.asarray(mask) != 0
    logp = np.zeros(len(logits), dtype=np.float64)
    vals = np.asarray(logits, dtype=np.float64)[legal]
    m = vals.max()
    s = np.exp(vals - m).sum()
    logp[legal] = (vals - m) - np.log(s)
    p = np.exp(logp[legal])
    ent = float(-(p * logp[legal]).sum())
    return logp, ent


def run_cache(lib, act_sizes, logits, mask, actions, old_lp):
    rows = logits.shape[0]
    a_total = logits.shape[1]
    sizes = np.ascontiguousarray(act_sizes, dtype=np.int32)
    logps = np.zeros((rows, a_total), dtype=np.float32)
    new_lp = np.zeros(rows, dtype=np.float32)
    entropy = np.zeros(rows, dtype=np.float32)
    imp = np.zeros(rows, dtype=np.float32)
    status = np.zeros(4, dtype=np.int32)
    rc = lib.puf_cat_shim_cache(
        rows, len(sizes), sizes,
        np.ascontiguousarray(logits, dtype=np.float32),
        np.ascontiguousarray(mask, dtype=np.uint8),
        np.ascontiguousarray(actions, dtype=np.float32),
        np.ascontiguousarray(old_lp, dtype=np.float32),
        logps, new_lp, entropy, imp, status)
    assert rc == 0, f"cache shim failed rc={rc} (-2 means guard band clobbered)"
    return logps, new_lp, entropy, imp, status


def run_sample(lib, act_sizes, logits, mask, uniforms, eps=0.0):
    rows = logits.shape[0]
    sizes = np.ascontiguousarray(act_sizes, dtype=np.int32)
    actions = np.zeros((rows, len(sizes)), dtype=np.float32)
    logp = np.zeros(rows, dtype=np.float32)
    status = np.zeros(4, dtype=np.int32)
    rc = lib.puf_cat_shim_sample(
        rows, len(sizes), sizes,
        np.ascontiguousarray(logits, dtype=np.float32),
        np.ascontiguousarray(mask, dtype=np.uint8),
        np.ascontiguousarray(uniforms, dtype=np.float32), ctypes.c_float(eps),
        actions, logp, status)
    assert rc == 0, f"sample shim failed rc={rc}"
    return actions, logp, status


def run_grad(lib, act_sizes, logps, mask, actions, dlogp, d_ent):
    rows = logps.shape[0]
    a_total = logps.shape[1]
    sizes = np.ascontiguousarray(act_sizes, dtype=np.int32)
    grad = np.zeros((rows, a_total), dtype=np.float32)
    rc = lib.puf_cat_shim_grad(
        rows, len(sizes), sizes,
        np.ascontiguousarray(logps, dtype=np.float32),
        np.ascontiguousarray(mask, dtype=np.uint8),
        np.ascontiguousarray(actions, dtype=np.float32),
        np.ascontiguousarray(dlogp, dtype=np.float32),
        ctypes.c_float(d_ent), grad)
    assert rc == 0, f"grad shim failed rc={rc}"
    return grad


def single_head(lib, width, mask, logits, action=0, old_lp=0.0):
    logits = quantize(lib, np.asarray(logits, dtype=np.float32).reshape(1, width))
    mask = np.asarray(mask, dtype=np.uint8).reshape(1, width)
    return run_cache(lib, [width], logits, mask,
                     np.array([[action]], dtype=np.float32),
                     np.array([old_lp], dtype=np.float32))


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1, 2, 31, 32, 33, 63, 64, 65, 155, 212, 257])
def test_logps_match_float64_reference(lib, width):
    rng = np.random.default_rng(width)
    logits = quantize(lib, rng.normal(0.0, 4.0, size=(1, width)))
    mask = (rng.random(width) > 0.4).astype(np.uint8).reshape(1, width)
    mask[0, 0] = 1  # first legal
    mask[0, -1] = 1  # last legal
    act = int(np.flatnonzero(mask[0])[0])
    logps, new_lp, entropy, imp, status = single_head(
        lib, width, mask, logits[0], action=act)
    assert status[0] == PUF_CAT_OK
    ref_logp, ref_ent = ref_head(logits[0], mask[0])
    legal = mask[0] != 0
    tol = 3e-2 if is_bf16(lib) else 2e-5
    assert np.allclose(logps[0][legal], ref_logp[legal], atol=tol, rtol=tol)
    assert np.allclose(entropy[0], ref_ent, atol=tol, rtol=tol)
    assert np.allclose(new_lp[0], ref_logp[act], atol=tol, rtol=tol)


def test_masked_entries_are_exactly_positive_zero(lib):
    width = 212
    rng = np.random.default_rng(7)
    logits = quantize(lib, rng.normal(0.0, 3.0, size=width))
    mask = np.ones(width, dtype=np.uint8)
    mask[5:200] = 0
    logps, _, _, _, status = single_head(lib, width, mask, logits, action=0)
    assert status[0] == PUF_CAT_OK
    masked = logps[0][mask == 0]
    assert np.all(masked == 0.0)
    assert not np.signbit(masked).any(), "masked logp must be +0, not -0"


def test_single_legal_category_is_certain(lib):
    width = 155
    logits = np.full(width, -3.0e4, dtype=np.float32)
    logits[77] = -2.5e4
    mask = np.zeros(width, dtype=np.uint8)
    mask[77] = 1
    logps, new_lp, entropy, imp, status = single_head(
        lib, width, mask, logits, action=77)
    assert status[0] == PUF_CAT_OK
    assert logps[0][77] == 0.0
    assert new_lp[0] == 0.0
    assert entropy[0] == 0.0 and not np.signbit(entropy[0])


def test_legal_below_minus_1e4_beats_masked_extremes(lib):
    """The old finite-sentinel scheme clamped masked logits to -1e4, which beat
    legal logits below it. Masked FLT_MAX/NaN/Inf must be ignored entirely."""
    width = 64
    logits = np.full(width, -3.0e4, dtype=np.float32)
    logits[10] = -1.1e4   # best legal
    logits[20] = -1.2e4
    mask = np.zeros(width, dtype=np.uint8)
    mask[10] = 1
    mask[20] = 1
    logits[0] = np.float32(3.4e38)
    logits[1] = np.float32("nan")
    logits[2] = np.float32("inf")
    logits[3] = np.float32("-inf")
    logits[40] = np.float32(1.0e30)
    logps, new_lp, entropy, imp, status = single_head(
        lib, width, mask, logits, action=10)
    assert status[0] == PUF_CAT_OK
    ref_logp, ref_ent = ref_head(quantize(lib, logits), mask)
    tol = 3e-2 if is_bf16(lib) else 1e-5
    assert abs(logps[0][10] - ref_logp[10]) < tol
    assert abs(logps[0][20] - ref_logp[20]) < tol
    assert logps[0][0] == 0.0 and logps[0][1] == 0.0
    assert np.isfinite(new_lp[0]) and np.isfinite(entropy[0])

    # Sampling must land on a legal category for every uniform draw.
    logits_row = quantize(lib, logits).reshape(1, width)
    mask_row = mask.reshape(1, width)
    for u in np.linspace(1e-6, 1.0, 64, dtype=np.float32):
        actions, _, status = run_sample(
            lib, [width], logits_row, mask_row, np.array([[u]], np.float32))
        assert status[0] == PUF_CAT_OK
        assert int(actions[0, 0]) in (10, 20)


def test_uniform_legal_shift_invariance(lib):
    """Integer logits in [-16, 16] shifted by integers that keep |value| <= 128
    stay exactly representable in float32 and bfloat16, so a legal-uniform shift
    must leave the normalized result bitwise unchanged. float32 also covers the
    large shifts."""
    width = 155
    rng = np.random.default_rng(11)
    base = np.clip(np.round(rng.normal(0.0, 6.0, size=width)), -16, 16)
    base = base.astype(np.float32)
    mask = (rng.random(width) > 0.3).astype(np.uint8)
    mask[0] = 1
    act = int(np.flatnonzero(mask)[0])
    shifts = [-96.0, -32.0, -1.0, 0.0, 1.0, 32.0, 96.0]
    if not is_bf16(lib):
        shifts += [-1.0e4, -8192.0, 8192.0, 1.0e4]
    ref = single_head(lib, width, mask, base, action=act)
    for shift in shifts:
        got = single_head(lib, width, mask,
                          (base + np.float32(shift)).astype(np.float32),
                          action=act)
        assert got[4][0] == PUF_CAT_OK
        legal = mask != 0
        assert np.array_equal(got[0][0][legal], ref[0][0][legal]), shift
        assert np.all(got[0][0][~legal] == 0.0)
        assert got[2][0] == ref[2][0], f"entropy changed under shift {shift}"
        assert got[1][0] == ref[1][0], f"selected logp changed under shift {shift}"


def test_heterogeneous_42_heads(lib):
    sizes = [155, 212, 1, 2, 31, 32, 33, 64, 65, 97] + [7] * 32
    assert len(sizes) == 42
    a_total = int(sum(sizes))
    rng = np.random.default_rng(42)
    rows = 5
    logits = quantize(lib, rng.normal(0.0, 3.0, size=(rows, a_total)))
    mask = (rng.random((rows, a_total)) > 0.35).astype(np.uint8)
    actions = np.zeros((rows, len(sizes)), dtype=np.float32)
    offset = 0
    for h, width in enumerate(sizes):
        for r in range(rows):
            mask[r, offset] = 1  # keep every head non-empty
            legal = np.flatnonzero(mask[r, offset:offset + width])
            actions[r, h] = float(legal[len(legal) // 2])
        offset += width
    old_lp = np.zeros(rows, dtype=np.float32)
    logps, new_lp, entropy, imp, status = run_cache(
        lib, sizes, logits, mask, actions, old_lp)
    assert status[0] == PUF_CAT_OK
    tol = 2e-1 if is_bf16(lib) else 1e-4
    for r in range(rows):
        offset = 0
        expect_lp = 0.0
        expect_ent = 0.0
        for h, width in enumerate(sizes):
            ref_logp, ref_ent = ref_head(
                logits[r, offset:offset + width], mask[r, offset:offset + width])
            got = logps[r, offset:offset + width]
            legal = mask[r, offset:offset + width] != 0
            assert np.allclose(got[legal], ref_logp[legal], atol=tol, rtol=tol)
            assert np.all(got[~legal] == 0.0)
            expect_lp += ref_logp[int(actions[r, h])]
            expect_ent += ref_ent
            offset += width
        assert abs(new_lp[r] - expect_lp) < max(tol, 1e-3 * abs(expect_lp) + tol)
        assert abs(entropy[r] - expect_ent) < max(tol, 1e-3 * expect_ent + tol)


def test_any_nonzero_mask_byte_is_legal(lib):
    """Environments may write any nonzero byte; the trainer's byte -> precision
    cast must keep those categories legal and zero bytes illegal."""
    width = 212
    rng = np.random.default_rng(3)
    logits = quantize(lib, rng.normal(0.0, 2.0, size=(1, width)))
    ones = (rng.random(width) > 0.5).astype(np.uint8).reshape(1, width)
    ones[0, 3] = 1
    varied = ones.copy()
    varied[0] = np.where(ones[0] != 0,
                         (np.arange(width) % 254 + 1).astype(np.uint8), 0)
    actions = np.array([[3]], dtype=np.float32)
    old = np.zeros(1, dtype=np.float32)
    a = run_cache(lib, [width], logits, ones, actions, old)
    b = run_cache(lib, [width], logits, varied, actions, old)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])


def test_empty_mask_is_fatal(lib):
    width = 155
    logits = quantize(lib, np.zeros(width, dtype=np.float32))
    mask = np.zeros(width, dtype=np.uint8)
    logps, new_lp, entropy, imp, status = single_head(lib, width, mask, logits)
    assert status[0] == PUF_CAT_ERR_EMPTY
    assert np.all(logps == 0.0) and np.isfinite(new_lp[0])
    assert imp[0] == 1.0 and entropy[0] == 0.0


def test_nonpositive_width_is_fatal(lib):
    logits = quantize(lib, np.zeros((1, 8), dtype=np.float32))
    mask = np.ones((1, 8), dtype=np.uint8)
    logps, new_lp, entropy, imp, status = run_cache(
        lib, [0, 8], logits, mask, np.zeros((1, 2), np.float32),
        np.zeros(1, np.float32))
    assert status[0] == PUF_CAT_ERR_WIDTH
    assert imp[0] == 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_legal_nonfinite_logit_is_fatal(lib, bad):
    width = 155
    logits = np.zeros(width, dtype=np.float32)
    logits[100] = np.float32(bad)
    mask = np.ones(width, dtype=np.uint8)
    logps, new_lp, entropy, imp, status = single_head(lib, width, mask, logits)
    assert status[0] == PUF_CAT_ERR_LEGAL_NONFINITE
    assert status[3] == 100
    assert np.all(np.isfinite(logps)) and imp[0] == 1.0


def test_epsilon_mixture_allows_individual_legal_negative_infinity(lib):
    logits = quantize(lib, np.array([[0.0, -np.inf]], dtype=np.float32))
    mask = np.ones((1, 2), dtype=np.uint8)
    actions, logp, status = run_sample(
        lib, [2], logits, mask, np.array([[0.9]], dtype=np.float32), eps=0.4)
    assert status[0] == PUF_CAT_OK
    assert actions[0, 0] == 1.0
    assert np.isfinite(logp[0])
    assert np.isclose(logp[0], np.log(0.2), atol=1e-6)


def test_epsilon_mixture_still_rejects_all_legal_negative_infinity(lib):
    logits = quantize(lib, np.full((1, 2), -np.inf, dtype=np.float32))
    mask = np.ones((1, 2), dtype=np.uint8)
    _, logp, status = run_sample(
        lib, [2], logits, mask, np.array([[0.9]], dtype=np.float32), eps=0.4)
    assert status[0] == PUF_CAT_ERR_LEGAL_NONFINITE
    assert np.isfinite(logp[0])


@pytest.mark.parametrize("action", [0.5, -1.0, 212.0, 1e9,
                                    float("nan"), float("inf")])
def test_invalid_action_is_fatal(lib, action):
    width = 212
    logits = quantize(lib, np.zeros(width, dtype=np.float32))
    mask = np.ones(width, dtype=np.uint8)
    logps, new_lp, entropy, imp, status = single_head(
        lib, width, mask, logits, action=action)
    assert status[0] == PUF_CAT_ERR_ACTION
    assert np.isfinite(new_lp[0]) and imp[0] == 1.0


def test_masked_action_is_fatal(lib):
    width = 64
    logits = quantize(lib, np.zeros(width, dtype=np.float32))
    mask = np.ones(width, dtype=np.uint8)
    mask[33] = 0
    logps, new_lp, entropy, imp, status = single_head(
        lib, width, mask, logits, action=33)
    assert status[0] == PUF_CAT_ERR_ACTION_ILLEGAL
    assert status[3] == 33


def test_nonfinite_old_logprob_is_fatal(lib):
    width = 32
    logits = quantize(lib, np.zeros(width, dtype=np.float32))
    mask = np.ones(width, dtype=np.uint8)
    logps, new_lp, entropy, imp, status = single_head(
        lib, width, mask, logits, action=1, old_lp=float("nan"))
    assert status[0] == PUF_CAT_ERR_OLD_LOGPROB
    assert np.isfinite(new_lp[0]) and imp[0] == 1.0


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
@pytest.mark.parametrize("width", [155, 212])
def test_sampling_never_selects_masked(lib, width):
    rng = np.random.default_rng(width + 1)
    logits = quantize(lib, rng.normal(0.0, 5.0, size=(1, width)))
    mask = (rng.random(width) > 0.7).astype(np.uint8).reshape(1, width)
    mask[0, width - 1] = 1
    illegal = np.flatnonzero(mask[0] == 0)
    logits[0, illegal] = np.float32(1.0e5)  # masked but dominant raw logits
    logits = quantize(lib, logits)
    uniforms = np.linspace(1e-7, 1.0, 512, dtype=np.float32).reshape(-1, 1)
    tiled_logits = np.repeat(logits, uniforms.shape[0], axis=0)
    tiled_mask = np.repeat(mask, uniforms.shape[0], axis=0)
    actions, logp, status = run_sample(
        lib, [width], tiled_logits, tiled_mask, uniforms)
    assert status[0] == PUF_CAT_OK
    chosen = actions[:, 0].astype(int)
    assert np.all(mask[0][chosen] != 0)
    assert np.all(np.isfinite(logp))


def test_sampling_matches_reference_distribution(lib):
    width = 97
    rng = np.random.default_rng(19)
    logits = quantize(lib, rng.normal(0.0, 1.5, size=(1, width)))
    mask = (rng.random(width) > 0.5).astype(np.uint8).reshape(1, width)
    mask[0, 0] = 1
    ref_logp, _ = ref_head(logits[0], mask[0])
    p = np.zeros(width)
    p[mask[0] != 0] = np.exp(ref_logp[mask[0] != 0])
    cdf = np.cumsum(p)
    n = 4096
    uniforms = ((np.arange(n) + 0.5) / n).astype(np.float32).reshape(-1, 1)
    actions, _, status = run_sample(
        lib, [width], np.repeat(logits, n, axis=0),
        np.repeat(mask, n, axis=0), uniforms)
    assert status[0] == PUF_CAT_OK
    expect = np.searchsorted(cdf, uniforms[:, 0], side="left")
    expect = np.minimum(expect, width - 1)
    # Compare positions among legal categories: masked gaps put neighbouring
    # legal categories many raw indices apart, and float rounding of a CDF
    # boundary may legitimately move a draw by one legal step.
    rank = np.cumsum(mask[0] != 0) - 1
    assert np.all(mask[0][actions[:, 0].astype(int)] != 0)
    drift = np.abs(rank[actions[:, 0].astype(int)] - rank[expect])
    assert np.all(drift <= 1), f"{(drift > 1).sum()} draws off by >1 legal step"


def test_epsilon_mixture_stays_legal(lib):
    width = 64
    rng = np.random.default_rng(5)
    logits = quantize(lib, rng.normal(0.0, 3.0, size=(1, width)))
    mask = np.zeros((1, width), dtype=np.uint8)
    mask[0, ::7] = 1
    n = 256
    uniforms = np.linspace(1e-6, 1.0, n, dtype=np.float32).reshape(-1, 1)
    actions, _, status = run_sample(
        lib, [width], np.repeat(logits, n, axis=0),
        np.repeat(mask, n, axis=0), uniforms, eps=0.4)
    assert status[0] == PUF_CAT_OK
    assert np.all(mask[0][actions[:, 0].astype(int)] != 0)


def test_unchanged_policy_ratio_is_exactly_one(lib):
    sizes = [155, 212, 3, 64]
    a_total = int(sum(sizes))
    rng = np.random.default_rng(2024)
    rows = 16
    logits = quantize(lib, rng.normal(0.0, 2.0, size=(rows, a_total)))
    mask = (rng.random((rows, a_total)) > 0.3).astype(np.uint8)
    offset = 0
    for width in sizes:
        mask[:, offset] = 1
        offset += width
    uniforms = rng.random((rows, len(sizes))).astype(np.float32)
    actions, old_lp, status = run_sample(lib, sizes, logits, mask, uniforms)
    assert status[0] == PUF_CAT_OK
    logps, new_lp, entropy, imp, status = run_cache(
        lib, sizes, logits, mask, actions, old_lp)
    assert status[0] == PUF_CAT_OK
    assert np.array_equal(new_lp, old_lp), "learner must reproduce rollout logp"
    assert np.all(imp == 1.0), "unchanged policy must give ratio exactly 1"


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------
def test_masked_gradients_are_exactly_positive_zero(lib):
    sizes = [155, 212]
    a_total = int(sum(sizes))
    rng = np.random.default_rng(8)
    rows = 4
    logits = quantize(lib, rng.normal(0.0, 2.0, size=(rows, a_total)))
    mask = (rng.random((rows, a_total)) > 0.5).astype(np.uint8)
    actions = np.zeros((rows, len(sizes)), dtype=np.float32)
    offset = 0
    for h, width in enumerate(sizes):
        mask[:, offset] = 1
        for r in range(rows):
            actions[r, h] = float(
                np.flatnonzero(mask[r, offset:offset + width])[0])
        offset += width
    logps, new_lp, entropy, imp, status = run_cache(
        lib, sizes, logits, mask, actions, np.zeros(rows, np.float32))
    assert status[0] == PUF_CAT_OK
    dlogp = rng.normal(size=rows).astype(np.float32)
    grad = run_grad(lib, sizes, logps, mask, actions, dlogp, -1e-3)
    masked = grad[mask == 0]
    assert np.all(masked == 0.0)
    assert not np.signbit(masked).any()
    assert np.all(np.isfinite(grad))

    # Legal gradient matches the analytic softmax gradient.
    tol = 2e-1 if is_bf16(lib) else 1e-4
    offset = 0
    for h, width in enumerate(sizes):
        for r in range(rows):
            ref_logp, ref_ent = ref_head(
                logits[r, offset:offset + width], mask[r, offset:offset + width])
            legal = mask[r, offset:offset + width] != 0
            p = np.exp(ref_logp) * legal
            act = int(actions[r, h])
            expect = ((np.arange(width) == act) - p) * dlogp[r] \
                + (-1e-3) * p * (-ref_ent - ref_logp)
            got = grad[r, offset:offset + width]
            assert np.allclose(got[legal], expect[legal], atol=tol, rtol=tol)
        offset += width


def test_random_reference_sweep(lib):
    rng = np.random.default_rng(1234)
    tol = 2e-1 if is_bf16(lib) else 1e-4
    for trial in range(12):
        num_heads = int(rng.integers(1, 8))
        sizes = [int(rng.integers(1, 260)) for _ in range(num_heads)]
        a_total = int(sum(sizes))
        rows = int(rng.integers(1, 9))
        logits = quantize(lib, rng.normal(0.0, rng.uniform(0.5, 20.0),
                                          size=(rows, a_total)))
        mask = (rng.random((rows, a_total)) > rng.uniform(0.1, 0.8)).astype(np.uint8)
        actions = np.zeros((rows, num_heads), dtype=np.float32)
        offset = 0
        for h, width in enumerate(sizes):
            for r in range(rows):
                mask[r, offset] = 1
                actions[r, h] = float(
                    np.flatnonzero(mask[r, offset:offset + width])[-1])
            offset += width
        logps, new_lp, entropy, imp, status = run_cache(
            lib, sizes, logits, mask, actions, np.zeros(rows, np.float32))
        assert status[0] == PUF_CAT_OK, f"trial {trial} status {status}"
        offset = 0
        for h, width in enumerate(sizes):
            for r in range(rows):
                ref_logp, ref_ent = ref_head(
                    logits[r, offset:offset + width],
                    mask[r, offset:offset + width])
                legal = mask[r, offset:offset + width] != 0
                got = logps[r, offset:offset + width]
                assert np.allclose(got[legal], ref_logp[legal], atol=tol, rtol=tol)
                assert np.all(got[~legal] == 0.0)
            offset += width


# --------------------------------------------------------------------------
# Integration: the runtime still compiles with the default sampler and with a
# legacy ENV_SAMPLE_LOGITS custom sampler (unchanged 10-argument ABI).
# --------------------------------------------------------------------------
def _nvcc_tu_flags():
    cuda_home = os.environ.get("CUDA_HOME") or os.path.dirname(
        os.path.dirname(NVCC))
    raylib = sorted(glob.glob(os.path.join(ROOT, "raylib-*")))
    if not raylib:
        pytest.skip("raylib headers missing (run ./build.sh <env> once first)")
    flags = [
        "-I" + ROOT, "-I" + os.path.join(ROOT, "src"),
        "-I" + os.path.join(ROOT, "vendor"),
        "-I" + os.path.join(ROOT, "ocean", "breakout"),
        "-I" + os.path.join(cuda_home, "include"),
        "-I" + os.path.join(cuda_home, "include", "cccl"),
        "-I" + os.path.join(raylib[-1], "include"),
    ]
    try:
        import nvidia.nccl
        flags.append("-I" + os.path.join(nvidia.nccl.__path__[0], "include"))
    except ImportError:
        pass
    return flags


def _compile_pufferl(tmp_path, extra_defines, tag, env="breakout"):
    obj = os.path.join(str(tmp_path), f"pufferl_{tag}.o")
    cmd = [
        NVCC, "-c", "-o", obj, os.path.join(ROOT, "src", "pufferl.cu"),
        "-std=c++17", "-O0", "-arch=" + os.environ.get("NVCC_ARCH", "native"),
        "--fmad=false", "-Xcompiler=-ffp-contract=off",
        "-Xcompiler=-Wno-narrowing", "--diag-suppress=2361",
        "-Xcompiler=-DPLATFORM_DESKTOP", "-Xcompiler=-fopenmp",
        f'-DENV_HEADER="ocean/{env}/{env}.h"',
        f"-DENV_NAME={env}", f'-DPUFFER_ENV_NAME="{env}"',
    ] + extra_defines + _nvcc_tu_flags()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-8000:]
    return obj


@pytest.mark.parametrize("precision", ["bf16", "float"])
def test_generic_runtime_compiles(tmp_path, precision):
    defines = ["-DPRECISION_FLOAT"] if precision == "float" else []
    _compile_pufferl(tmp_path, defines, "generic_" + precision)


def test_legacy_custom_sampler_hook_still_compiles(tmp_path):
    defines = [
        '-DENV_ACTION_KERNEL_HEADER="../tests/legacy_sample_hook.cuh"',
        "-DENV_SAMPLE_LOGITS=legacy_sample_logits",
    ]
    _compile_pufferl(tmp_path, defines, "legacy_hook")


def test_nonfinite_continuous_custom_sampler_hook_still_compiles(tmp_path):
    defines = [
        '-DENV_ACTION_KERNEL_HEADER="../tests/nonfinite_sample_hook.cuh"',
        "-DENV_SAMPLE_LOGITS=nonfinite_sample_logits",
    ]
    _compile_pufferl(
        tmp_path, defines, "nonfinite_continuous_hook",
        env="squared_continuous")
