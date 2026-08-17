from dataclasses import dataclass

import torch


GROUPED_PPO_CAPABILITY = "pufferlib.grouped_ppo.v1"
GROUPED_PPO_CAPABILITY_VERSION = 1
PPO_CLIP_MODES = ("joint", "per_head", "group_sum", "group_mean")

GROUPED_PPO_METRIC_KEYS = (
    "ppo_active_stochastic_heads",
    "ppo_active_stochastic_groups",
    "ppo_optimized_kl",
    "ppo_optimized_clipfrac",
    "ppo_optimized_entropy",
    "ppo_joint_old_kl",
    "ppo_joint_kl",
    "ppo_joint_clipfrac",
)


def validate_ppo_clip_mode(mode):
    if mode not in PPO_CLIP_MODES:
        choices = "|".join(PPO_CLIP_MODES)
        raise ValueError(f"train.ppo_clip_mode must be one of {choices}, got {mode!r}")
    return mode


@dataclass(frozen=True)
class FactorizedPPOResult:
    policy_loss: torch.Tensor
    joint_ratio: torch.Tensor
    metrics: dict


def _masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def factorized_ppo_loss(
        old_logprobs, new_logprobs, entropies, advantages, head_used,
        legal_counts, head_to_group, clip_coef, mode="joint", trainable=None):
    """Compute PPO clipping over joint, head, or within-transition group units.

    All tensors except ``head_to_group`` have leading transition dimensions.
    The last dimension of per-head tensors is the action-head dimension.
    """
    mode = validate_ppo_clip_mode(mode)
    if old_logprobs.shape != new_logprobs.shape:
        raise ValueError("old_logprobs and new_logprobs must have identical shapes")
    if old_logprobs.shape != entropies.shape:
        raise ValueError("entropies must have the same per-head shape as logprobs")
    if head_used.shape != old_logprobs.shape:
        raise ValueError("head_used must have the same per-head shape as logprobs")
    if legal_counts.shape != old_logprobs.shape:
        raise ValueError("legal_counts must have the same per-head shape as logprobs")

    num_heads = old_logprobs.shape[-1]
    if head_to_group is None:
        if mode != "joint":
            raise ValueError(f"ppo_clip_mode={mode!r} requires a grouped PPO adapter")
        head_to_group = torch.arange(num_heads, device=old_logprobs.device)
    else:
        head_to_group = torch.as_tensor(
            head_to_group, dtype=torch.long, device=old_logprobs.device)
        if head_to_group.shape != (num_heads,):
            raise ValueError("head_to_group must contain exactly one group per action head")
        if num_heads and (head_to_group.min() < 0):
            raise ValueError("head_to_group values must be non-negative")

    transition_shape = old_logprobs.shape[:-1]
    if trainable is None:
        trainable = torch.ones(transition_shape, dtype=torch.bool,
            device=old_logprobs.device)
    else:
        trainable = torch.as_tensor(
            trainable, dtype=torch.bool, device=old_logprobs.device)
        if trainable.shape != transition_shape:
            raise ValueError("trainable must have the logprob transition shape")

    active = head_used.bool() & (legal_counts > 1) & trainable.unsqueeze(-1)
    logratios = torch.where(
        active, new_logprobs - old_logprobs,
        torch.zeros((), dtype=old_logprobs.dtype, device=old_logprobs.device))
    safe_entropies = torch.where(
        active, entropies,
        torch.zeros((), dtype=entropies.dtype, device=entropies.device))
    joint_logratio = logratios.sum(-1)
    joint_ratio = torch.where(
        trainable, joint_logratio.clamp(-20, 20).exp(),
        torch.ones_like(joint_logratio))

    groups = torch.unique(head_to_group, sorted=True)
    group_active = torch.stack([
        active[..., head_to_group == group].any(-1) for group in groups
    ], dim=-1) if groups.numel() else active[..., :0]
    active_heads = active.sum(-1)
    active_groups = group_active.sum(-1)

    if mode == "joint":
        unit_logratios = joint_logratio.unsqueeze(-1)
        unit_active = trainable.unsqueeze(-1)
    elif mode == "per_head":
        unit_logratios = logratios
        unit_active = active
    else:
        grouped = []
        for group in groups:
            members = head_to_group == group
            member_active = active[..., members]
            value = (logratios[..., members] * member_active).sum(-1)
            if mode == "group_mean":
                value = value / member_active.sum(-1).clamp_min(1)
            grouped.append(value)
        unit_logratios = torch.stack(grouped, dim=-1)
        unit_active = group_active

    ratios = unit_logratios.clamp(-20, 20).exp()
    clipped = ratios.clamp(1 - clip_coef, 1 + clip_coef)
    unit_pg = torch.maximum(
        -advantages.unsqueeze(-1) * ratios,
        -advantages.unsqueeze(-1) * clipped,
    )
    transition_pg = (
        unit_pg * unit_active
    ).sum(-1) / unit_active.sum(-1).clamp_min(1)
    policy_loss = _masked_mean(transition_pg, trainable)

    optimized_kl = (ratios - 1) - unit_logratios
    optimized_clip = (ratios - 1).abs() > clip_coef
    joint_kl = (joint_ratio - 1) - joint_logratio
    transition_optimized_kl = (
        optimized_kl * unit_active
    ).sum(-1) / unit_active.sum(-1).clamp_min(1)
    transition_optimized_clip = (
        optimized_clip * unit_active
    ).sum(-1) / unit_active.sum(-1).clamp_min(1)
    transition_entropy = safe_entropies.sum(-1) / active.sum(-1).clamp_min(1)
    metrics = {
        "ppo_active_stochastic_heads": _masked_mean(
            active_heads.to(old_logprobs.dtype), trainable),
        "ppo_active_stochastic_groups": _masked_mean(
            active_groups.to(old_logprobs.dtype), trainable),
        "ppo_optimized_kl": _masked_mean(
            transition_optimized_kl, trainable),
        "ppo_optimized_clipfrac": _masked_mean(
            transition_optimized_clip.to(old_logprobs.dtype), trainable),
        "ppo_optimized_entropy": _masked_mean(
            transition_entropy, trainable),
        "ppo_joint_old_kl": _masked_mean(-joint_logratio, trainable),
        "ppo_joint_kl": _masked_mean(joint_kl, trainable),
        "ppo_joint_clipfrac": _masked_mean(
            ((joint_ratio - 1).abs() > clip_coef).to(old_logprobs.dtype),
            trainable),
    }
    return FactorizedPPOResult(policy_loss, joint_ratio, metrics)
