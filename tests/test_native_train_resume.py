from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src" / "pufferl.cu"


def _function_body(name):
    source = SOURCE.read_text()
    signature = f"{name}("
    start = source.index(signature)
    body_start = source.index("{", start)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1:index]
    raise AssertionError(f"unterminated function: {name}")


def _block_after(source, condition):
    start = source.index(condition)
    body_start = source.index("{", start)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1:index]
    raise AssertionError(f"unterminated block: {condition}")


def test_resume_load_precedes_selfplay_and_rollout():
    train = _function_body("run_train")

    create = train.index("create_pufferl(ini, ctx)")
    resolve = train.index('puf_checkpoint_path_key(\n        ini, "load_model_path"')
    load = train.index("pufferl_load_policy(pufferl, 0, load_path)")
    selfplay = train.index("Selfplay selfplay = {0}")
    rollout = min(train.index("rollout_start(pufferl"), train.index("rollouts(pufferl)"))

    assert create < resolve < load < selfplay < rollout


def test_resume_saves_exact_loaded_weights_before_update():
    train = _function_body("run_train")
    load = train.index("pufferl_load_policy(pufferl, 0, load_path)")
    save = train.index("puf_save_weights(pufferl, initial_checkpoint)")
    rollout = min(train.index("rollout_start(pufferl"), train.index("rollouts(pufferl)"))

    assert load < save < rollout
    assert '"%s/%016ld.bin", checkpoint_dir, pufferl->global_step' in train

    load_policy = _function_body("pufferl_load_policy")
    save_weights = _function_body("puf_save_weights")
    assert "puf_load_weights_into(pol->master_weights, pol->param" in load_policy
    assert "Float mw = p->policies[0].master_weights" in save_weights


def test_resume_does_not_restore_optimizer_state():
    train = _function_body("run_train")
    resume_setup = train[
        train.index("create_pufferl(ini, ctx)"):
        train.index("Selfplay selfplay = {0}")
    ]
    load_policy = _function_body("pufferl_load_policy")
    create = _function_body("create_pufferl")

    assert "muon" not in resume_setup
    assert "muon" not in load_policy
    assert "muon_init(" in create
    assert "cudaMemset(pufferl->muon.mb.data, 0" in create


def test_scratch_training_keeps_initial_checkpoint_behavior():
    train = _function_body("run_train")
    load_block = _block_after(train, "if (load_path)")
    initial_save_block = _block_after(train, "if (load_path || use_selfplay)")

    assert "pufferl_load_policy(pufferl, 0, load_path)" in load_block
    assert "puf_save_weights(pufferl, initial_checkpoint)" in initial_save_block
    assert train.count("puf_save_weights(pufferl, initial_checkpoint)") == 1
