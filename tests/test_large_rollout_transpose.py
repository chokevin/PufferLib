from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_large_rollout_transposes_use_64_bit_grid_stride_indexing():
    source = (ROOT / "src" / "pufferl.cu").read_text()
    transposes = source[
        source.index("// Transpose (A, B, C)") : source.index(
            "// Cosine decay"
        )
    ]
    launches = source[
        source.index("static void train_epoch_gpu(") : source.index(
            "void train_impl("
        )
    ]

    assert 720 * 1000 * 5207 > 2**31 - 1
    assert 720 * 1000 * 3295 > 2**31 - 1
    assert transposes.count("int64_t total =") == 4
    assert transposes.count("idx += stride") == 4
    assert "large_grid_size(\n            (int64_t)T * train_B * obs_size)" in launches
    assert "large_grid_size(\n            (int64_t)T * train_B * mask_c)" in launches
    assert "rollout-to-train transpose launch failed" in launches
