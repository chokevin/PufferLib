import configparser
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pufferlib.ppo import (
    GROUPED_PPO_CAPABILITY,
    GROUPED_PPO_CAPABILITY_VERSION,
    GROUPED_PPO_METRIC_KEYS,
    PPO_CLIP_MODES,
    factorized_ppo_loss,
    validate_ppo_clip_mode,
)


def _inputs(new, *, used=None, legal=None, groups=None, advantages=None):
    new = torch.as_tensor(new, dtype=torch.float64)
    old = torch.zeros_like(new)
    entropy = torch.ones_like(new)
    used = torch.ones_like(new, dtype=torch.bool) if used is None else torch.as_tensor(used)
    legal = torch.full_like(new, 2, dtype=torch.long) if legal is None else torch.as_tensor(legal)
    groups = torch.arange(new.shape[-1]) if groups is None else torch.as_tensor(groups)
    advantages = torch.ones(new.shape[:-1], dtype=new.dtype) \
        if advantages is None else torch.tensor(advantages, dtype=new.dtype)
    return old, new, entropy, advantages, used, legal, groups


def _clipped_loss(logratio, advantage=1.0, clip=0.2):
    ratio = math.exp(logratio)
    return max(-advantage * ratio, -advantage * min(max(ratio, 1 - clip), 1 + clip))


def test_mode_default_and_exact_values():
    config = configparser.ConfigParser()
    config.read(Path(__file__).parents[1] / "config" / "default.ini")
    assert config["train"]["ppo_clip_mode"] == "joint"
    assert PPO_CLIP_MODES == ("joint", "per_head", "group_sum", "group_mean")
    for mode in PPO_CLIP_MODES:
        assert validate_ppo_clip_mode(mode) == mode
    with pytest.raises(ValueError, match="joint\\|per_head\\|group_sum\\|group_mean"):
        validate_ppo_clip_mode("sum")


@pytest.mark.parametrize("mode, unit_logratios", [
    ("joint", [0.5]),
    ("per_head", [0.3, -0.2, 0.4]),
    ("group_sum", [0.1, 0.4]),
    ("group_mean", [0.05, 0.4]),
])
def test_exact_objective_math(mode, unit_logratios):
    args = _inputs([[0.3, -0.2, 0.4]], groups=[0, 0, 1])
    result = factorized_ppo_loss(*args, clip_coef=0.2, mode=mode)
    expected = sum(_clipped_loss(value) for value in unit_logratios) \
        / len(unit_logratios)
    assert result.policy_loss.item() == pytest.approx(expected)


def test_consumed_and_deterministic_heads_are_excluded():
    args = _inputs(
        [[0.1, 9.0, -8.0, 0.2]],
        used=[[True, False, True, True]],
        legal=[[2, 2, 1, 3]],
        groups=[0, 0, 1, 1],
    )
    result = factorized_ppo_loss(*args, clip_coef=0.2, mode="group_sum")
    assert result.joint_ratio.item() == pytest.approx(math.exp(0.3))
    assert result.metrics["ppo_active_stochastic_heads"].item() == 2
    assert result.metrics["ppo_active_stochastic_groups"].item() == 2


def test_inactive_nonfinite_heads_do_not_leak():
    args = _inputs(
        [[0.1, float("nan"), float("inf")]],
        used=[[True, False, True]],
        legal=[[2, 2, 1]],
        groups=[0, 0, 1],
    )
    result = factorized_ppo_loss(*args, clip_coef=0.2, mode="group_sum")
    assert result.joint_ratio.item() == pytest.approx(math.exp(0.1))
    assert torch.isfinite(result.policy_loss)
    assert all(torch.isfinite(value) for value in result.metrics.values())


def test_groups_never_cross_original_transitions():
    head_to_group = [-1] * 252
    groups = []
    for i in range(32):
        group = [i, 42 + i, 74 + i, 106 + i, 138 + i, 170 + i]
        groups.append(group)
    for j in range(10):
        group = [32 + j, 202 + j, 212 + j, 222 + j, 232 + j, 242 + j]
        groups.append(group)
    for group_id, members in enumerate(groups):
        for head in members:
            assert head_to_group[head] == -1
            head_to_group[head] = group_id
    assert len(groups) == 42
    assert sorted(head for group in groups for head in group) == list(range(252))

    new = torch.zeros((2, 252), dtype=torch.float64)
    new[0, groups[0]] = 0.1
    new[1, groups[0]] = -0.2
    used = torch.zeros_like(new, dtype=torch.bool)
    used[:, groups[0]] = True
    args = _inputs(new, used=used, groups=head_to_group)
    result = factorized_ppo_loss(*args, clip_coef=0.2, mode="group_sum")
    expected = (_clipped_loss(0.6) + _clipped_loss(-1.2)) / 2
    assert result.policy_loss.item() == pytest.approx(expected)


def test_frozen_opponent_transitions_are_never_optimized():
    new = torch.tensor(
        [[0.1, 0.2], [10.0, 10.0]], dtype=torch.float64,
        requires_grad=True)
    args = _inputs(
        new,
        groups=[0, 0],
        advantages=[1.0, -100.0],
    )
    result = factorized_ppo_loss(
        *args, clip_coef=0.2, mode="group_sum",
        trainable=torch.tensor([True, False]))
    assert result.policy_loss.item() == pytest.approx(_clipped_loss(0.3))
    assert result.joint_ratio.tolist() == pytest.approx([math.exp(0.3), 1.0])
    result.policy_loss.backward()
    assert torch.count_nonzero(new.grad[1]) == 0


def test_joint_matches_legacy_summed_logratio():
    args = _inputs(
        [[0.1, -0.05, 0.2], [-0.1, 0.3, -0.2]],
        advantages=[0.7, -1.2],
    )
    result = factorized_ppo_loss(*args, clip_coef=0.2, mode="joint")
    expected = sum(
        _clipped_loss(sum(row), advantage, 0.2)
        for row, advantage in zip(args[1].tolist(), args[3].tolist())
    ) / 2
    assert result.policy_loss.item() == pytest.approx(expected)


def test_metrics_are_stable_finite_epoch_means():
    args = _inputs(
        [[0.1, 0.2, 5.0], [0.0, -0.1, -4.0]],
        used=[[True, True, False], [True, False, False]],
        groups=[0, 0, 1],
    )
    result = factorized_ppo_loss(*args, clip_coef=0.2, mode="group_mean")
    assert tuple(result.metrics) == GROUPED_PPO_METRIC_KEYS
    assert result.metrics["ppo_active_stochastic_heads"].item() == 1.5
    assert result.metrics["ppo_active_stochastic_groups"].item() == 1.0
    assert all(torch.isfinite(value) for value in result.metrics.values())
    assert GROUPED_PPO_CAPABILITY == "pufferlib.grouped_ppo.v1"
    assert GROUPED_PPO_CAPABILITY_VERSION == 1


def test_extreme_logratios_keep_metrics_finite():
    args = _inputs([[1000.0, -1000.0]], groups=[0, 1])
    result = factorized_ppo_loss(*args, clip_coef=0.2, mode="per_head")
    assert torch.isfinite(result.policy_loss)
    assert torch.isfinite(result.joint_ratio).all()
    assert all(torch.isfinite(value) for value in result.metrics.values())


def test_non_joint_requires_valid_adapter():
    args = list(_inputs([[0.1, 0.2]]))
    args[-1] = None
    with pytest.raises(ValueError, match="requires a grouped PPO adapter"):
        factorized_ppo_loss(*args, clip_coef=0.2, mode="per_head")

    args[-1] = torch.tensor([0])
    with pytest.raises(ValueError, match="exactly one group"):
        factorized_ppo_loss(*args, clip_coef=0.2, mode="group_sum")
