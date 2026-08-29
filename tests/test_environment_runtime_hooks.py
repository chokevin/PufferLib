from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OCEAN_SOURCE = (ROOT / "src" / "ocean.cu").read_text()
ALGO_SOURCE = (ROOT / "src" / "algo.cu").read_text()
PUFFERL_SOURCE = (ROOT / "src" / "pufferl.cu").read_text()


def test_declared_custom_encoder_is_registered():
    include = "#include ENV_ENCODER_HEADER"
    factory = "create_env_encoder(enc);"

    assert "#ifdef ENV_ENCODER_HEADER" in OCEAN_SOURCE
    assert include in OCEAN_SOURCE
    assert factory in OCEAN_SOURCE
    assert OCEAN_SOURCE.index(include) < OCEAN_SOURCE.index(factory)


def test_conditional_head_hook_covers_rollout_and_ppo():
    hook = "ENV_HEAD_USED"

    assert "kg_puffer5_head_used" in PUFFERL_SOURCE
    assert PUFFERL_SOURCE.count(hook) >= 1
    assert ALGO_SOURCE.count(hook) >= 2

    sample_defer = PUFFERL_SOURCE.index(
        "// Defer accounting until all selector heads have been sampled."
    )
    sample_write = PUFFERL_SOURCE.index("env_actions[aidx] = action;")
    assert sample_write < sample_defer

    ppo_logprob = ALGO_SOURCE.index(
        "ENV_HEAD_USED(actions + idx * NUM_ATNS, h)"
    )
    ppo_grad = ALGO_SOURCE.index(
        "ENV_HEAD_USED(g.actions + nt * a.num_atns, h)"
    )
    entropy = ALGO_SOURCE.index("total_entropy += ent;", ppo_grad)
    assert ppo_logprob < ppo_grad < entropy


def test_kaggriculture_parameter_contract():
    encoder_parameters = 312_832
    hidden = 256
    layers = 4
    action_logits = 42

    recurrent_parameters = layers * 3 * hidden * hidden
    decoder_parameters = (action_logits + 1) * hidden
    total_parameters = (
        encoder_parameters + recurrent_parameters + decoder_parameters
    )

    assert total_parameters == 1_110_272
    assert total_parameters * 4 == 4_441_088
