from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_train_loads_configured_primary_checkpoint():
    source = (ROOT / "src" / "pufferl.cu").read_text()
    run_train = source[
        source.index("TrainResult run_train(") : source.index(
            "TrainResult launch_train("
        )
    ]

    create = run_train.index("create_pufferl(ini, ctx)")
    load = run_train.index("pufferl_load_policy(pufferl, 0, primary_model)")
    rollout = run_train.index("rollout_start(pufferl")

    assert create < load < rollout
    assert 'puf_checkpoint_path_key(\n        ini, "load_model_path"' in run_train
