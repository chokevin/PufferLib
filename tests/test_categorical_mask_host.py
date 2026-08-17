"""Runs tests/categorical_host_exec.cc: the production categorical primitive
(src/categorical.cuh) executed on the host through the 32-lane warp emulator in
tests/cuda_stub/. This is the coverage that works on machines without CUDA;
tests/test_categorical_mask.py runs the same semantics on a real GPU.

Run: pytest -q tests/test_categorical_mask_host.py
"""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "categorical_host_exec.cc")
OUT_DIR = os.path.join(ROOT, "build")

CXX = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") \
    or shutil.which("g++")
if CXX is None:
    pytest.skip("no host C++ compiler available", allow_module_level=True)


def build(float_precision):
    os.makedirs(OUT_DIR, exist_ok=True)
    name = "categorical_host_float" if float_precision else "categorical_host_bf16"
    out = os.path.join(OUT_DIR, name)
    cmd = [
        CXX, "-std=c++17", "-O1", "-Wall",
        "-I" + os.path.join(HERE, "cuda_stub"),
        "-o", out, SRC,
    ]
    if float_precision:
        cmd.append("-DPRECISION_FLOAT")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"build failed:\n{proc.stderr}"
    return out


@pytest.mark.parametrize("float_precision", [True, False], ids=["float", "bf16"])
def test_categorical_primitive_host_semantics(float_precision):
    binary = build(float_precision)
    proc = subprocess.run([binary], capture_output=True, text=True, timeout=900)
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ", 0 failures" in proc.stdout
    checks = int(proc.stdout.strip().split("\n")[-1].split(" ")[0])
    assert checks > 1000, f"expected a broad sweep, ran only {checks} checks"


@pytest.mark.parametrize("float_precision", [True], ids=["float"])
def test_status_check_aborts_before_consumer(float_precision):
    """puf_cat_status_check must surface the row/head context and abort rather
    than let a failing row reach an environment step or the optimizer."""
    binary = build(float_precision)
    proc = subprocess.run([binary, "abort"], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode != 0 and proc.returncode != 2, proc.stdout
    assert "fatal action-mask error in unit test consumer" in proc.stderr
    assert "stored action is masked out" in proc.stderr
    assert "row 11" in proc.stderr and "head 5" in proc.stderr
    assert "detail 93" in proc.stderr
