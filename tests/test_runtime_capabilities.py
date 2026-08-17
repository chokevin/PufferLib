"""Host coverage for environment decoder and strict checkpoint capabilities."""

import os
import re
import shutil
import subprocess
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
CXX = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") \
    or shutil.which("g++")
if CXX is None:
    pytest.skip("no host C++ compiler available", allow_module_level=True)


def read(name):
    with open(os.path.join(SRC, name), "r") as handle:
        return handle.read()


def order(text, *needles):
    last = -1
    for needle in needles:
        idx = text.find(needle, last + 1)
        assert idx > last, f"expected {needle!r} after position {last}"
        last = idx


def compile_program(directory, name, source, extra_args=()):
    source_path = os.path.join(directory, name + ".cc")
    binary_path = os.path.join(directory, name)
    with open(source_path, "w") as handle:
        handle.write(source)
    proc = subprocess.run(
        [CXX, "-std=c++17", "-Wall", "-Werror", "-I" + SRC,
         *extra_args, source_path, "-o", binary_path],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return binary_path


def test_capability_markers_precede_environment_include_chain():
    pufferl = read("pufferl.cu")
    algo = read("algo.cu")
    ocean = read("ocean.cu")
    decoder = read("custom_decoder.cuh")

    strict_checkpoint_marker = "#define PUF_STRICT_CHECKPOINT_SIZE_V1 1"
    assert pufferl.count(strict_checkpoint_marker) == 1
    order(pufferl,
          strict_checkpoint_marker,
          '#include "checkpoint.h"',
          "#include ENV_HEADER",
          '#include "algo.cu"')
    order(algo, '#include "ocean.cu"')
    order(ocean,
          '#include "custom_decoder.cuh"',
          "#include ENV_ENCODER_HEADER")
    order(decoder,
          "#define PUF_ENV_DECODER_HEADER_V1 1",
          "#include ENV_DECODER_HEADER",
          "static void create_custom_decoder")


def test_checkpoint_loader_checks_size_and_complete_read():
    loader = read("checkpoint.h")
    pufferl = read("pufferl.cu")

    assert "fstat(fileno(fp), &info) == 0" in loader
    assert "info.st_size == nbytes" in loader
    assert "nread == expected && !ferror(fp)" in loader
    body = re.search(
        r"void puf_load_weights_into\(.*?\n\}",
        pufferl,
        flags=re.S,
    ).group(0)
    order(body,
          "puf_read_checkpoint_exact(path, nbytes)",
          "cudaMemcpy(dst.data, buf, nbytes, cudaMemcpyHostToDevice)")


DECODER_HARNESS = r"""
#include <cstring>

struct Decoder {
    int kind;
};

#ifdef PUFFER_NETHACK
static void create_nethack_decoder(Decoder* dec) {
    dec->kind = 2;
}
#endif

#include "custom_decoder.cuh"

int main(int argc, char** argv) {
    Decoder dec = {1};
    create_custom_decoder(argc > 1 ? argv[1] : "generic", &dec);
    return dec.kind;
}
"""


@pytest.mark.parametrize(
    ("defines", "env_name", "expected"),
    [
        ([], "generic", 1),
        (["-DPUFFER_NETHACK"], "generic", 1),
        (["-DPUFFER_NETHACK"], "nethack", 2),
    ],
    ids=["generic", "nethack-generic", "nethack"],
)
def test_existing_decoder_fallbacks(defines, env_name, expected):
    with tempfile.TemporaryDirectory() as tmp:
        binary = compile_program(tmp, "decoder_fallback", DECODER_HARNESS, defines)
        proc = subprocess.run([binary, env_name], capture_output=True, text=True)
        assert proc.returncode == expected, proc.stdout + proc.stderr


def test_configured_environment_decoder_is_included_and_invoked_first():
    with tempfile.TemporaryDirectory() as tmp:
        header = os.path.join(tmp, "env_decoder.h")
        with open(header, "w") as handle:
            handle.write(r"""
#ifndef PUF_ENV_DECODER_HEADER_V1
#error "decoder capability marker must precede ENV_DECODER_HEADER"
#endif
static void create_env_decoder(Decoder* dec) {
    dec->kind = 3;
}
""")
        binary = compile_program(
            tmp,
            "env_decoder",
            DECODER_HARNESS,
            [
                "-DPUFFER_NETHACK",
                '-DENV_DECODER_HEADER="env_decoder.h"',
                "-I" + tmp,
            ],
        )
        for env_name in ("generic", "nethack"):
            proc = subprocess.run(
                [binary, env_name], capture_output=True, text=True
            )
            assert proc.returncode == 3, proc.stdout + proc.stderr


CHECKPOINT_HARNESS = r"""
#include <cstdlib>
#include <cstring>
#include "checkpoint.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }
    char* buf = puf_read_checkpoint_exact(argv[1], 4);
    bool valid = std::memcmp(buf, "PUFF", 4) == 0;
    std::free(buf);
    return valid ? 0 : 3;
}
"""


def test_checkpoint_exact_size_succeeds():
    with tempfile.TemporaryDirectory() as tmp:
        binary = compile_program(tmp, "checkpoint", CHECKPOINT_HARNESS)
        checkpoint = os.path.join(tmp, "exact.bin")
        with open(checkpoint, "wb") as handle:
            handle.write(b"PUFF")
        proc = subprocess.run([binary, checkpoint], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize(
    ("name", "contents"),
    [("short", b"PUF"), ("trailing", b"PUFF!")],
)
def test_checkpoint_wrong_size_is_rejected(name, contents):
    with tempfile.TemporaryDirectory() as tmp:
        binary = compile_program(tmp, "checkpoint", CHECKPOINT_HARNESS)
        checkpoint = os.path.join(tmp, name + ".bin")
        with open(checkpoint, "wb") as handle:
            handle.write(contents)
        proc = subprocess.run([binary, checkpoint], capture_output=True, text=True)
        assert proc.returncode != 0 and proc.returncode != 3
        assert "weights file size mismatch" in proc.stderr
