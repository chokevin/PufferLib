"""Host type-check of the modified PPO CUDA kernels.

nvcc is not always available (and a GPU never is in CI here), so this test
extracts the relevant kernel source verbatim from src/algo.cu and
src/pufferl.cu, compiles it against small CUDA stubs (tests/ppo_kernel_stub.h)
with a host C++ compiler, and fails on any syntax or type error. Kernel launch
syntax (<<<>>>) is nvcc-only and is therefore excluded from the extraction.

It runs five configurations so the generic, Nethack, canonical adapter, and
legacy Kaggriculture head-used branches are covered, including NUM_ATNS=252.
"""

import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")

# (file, start marker, end marker) — verbatim slices of production source.
SLICES = [
    ("src/pufferl.cu", "static size_t checked_size_product(",
     "// Exclusive env:"),
    ("src/algo.cu", "// Core loss channels.", "#ifdef PUFFER_NETHACK"),
    ("src/algo.cu", "// Per-env from ENV_HEADER",
     "// Discrete only. mask is always present"),
    ("src/algo.cu", "// Discrete only. mask is always present",
     "Prec arch_forward_train("),
    ("src/algo.cu", "// Deterministic block tree-reduce with max.",
     "void ppo_loss_fwd_bwd("),
    ("src/pufferl.cu", "__global__ void sample_logits(",
     "#ifdef ENV_ACTION_KERNEL_HEADER"),
]

CONFIGS = [
    ("generic", [], 4),
    ("nethack", ["-DPUFFER_NETHACK"], 4),
    ("head-adapter", ["-DPUFFER_PROVIDES_PPO_HEAD_USED"], 4),
    ("kg", ["-DKG_HEAD_GATING_SELECTOR_VALUES"], 4),
    ("kg-252", ["-DKG_HEAD_GATING_SELECTOR_VALUES"], 252),
]


def _extract(num_atns=4):
    parts = []
    if num_atns != 4:
        parts.extend([
            f"#define NUM_ATNS {num_atns}",
            "#define ACT_SIZES {" + ",".join(["2"] * num_atns) + "}",
        ])
    parts.extend(['#include "ppo_kernel_stub.h"',
             '#include "ppo_kernel_env_hooks.h"',
             '#include "../src/ppo_group.h"'])
    for path, start, end in SLICES:
        text = open(os.path.join(ROOT, path)).read()
        a = text.index(start)
        b = text.index(end, a)
        assert b > a, f"empty slice for {path}: {start}"
        parts.append(text[a:b])
    return "\n\n".join(parts)


@pytest.mark.parametrize(
    "name,extra,num_atns", CONFIGS, ids=[cfg[0] for cfg in CONFIGS])
def test_modified_ppo_kernels_type_check(tmp_path, name, extra, num_atns):
    cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        pytest.skip("no host C++ compiler available")
    src = tmp_path / "extracted.cc"
    src.write_text(_extract(num_atns))
    cmd = [cxx, "-std=c++17", "-fsyntax-only", "-Wall", "-Wno-unused-function",
           "-I", TESTS] + extra + [str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_extraction_covers_the_changed_kernels():
    text = _extract()
    for symbol in ["cache_imp_and_v", "ppo_loss_compute", "ppo_loss_reduce",
                   "sample_logits", "ppo_group_row_apply", "ppo_head_consumed",
                   "ppo_mask_legal_count_prec", "ppo_block_reduce_max",
                   "checked_size_product", "checked_launch_product"]:
        assert symbol in text, f"{symbol} not covered by the syntax harness"
    assert "code == PPO_ERR_NONE && consumed" in text
    assert "if (num_groups > 0 && n_active_heads == 0)" in text
    assert "if (!joint && !isfinite(thread_loss))" in text
