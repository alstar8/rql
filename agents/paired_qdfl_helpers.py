"""Shared helpers for paired QuantizedDFLRQL9 teacher + token-student agents.

Used by ``discrete_ar_qdfl_distill`` and ``discrete_diffusion_qdfl_distill``.
Keeps Python-side orchestration utilities out of the individual agent modules.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

import jax
import jax.numpy as jnp
import numpy as np


def prefix_info(info: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """Prefix every info key (for merging teacher/student diagnostics)."""
    return {f"{prefix}{key}": value for key, value in info.items()}


def merge_infos(*infos: Mapping[str, Any]) -> Dict[str, Any]:
    """Left-to-right merge of info dicts (later keys overwrite)."""
    out: Dict[str, Any] = {}
    for info in infos:
        out.update(info)
    return out


def tree_leaves_allclose(a, b, rtol: float = 0.0, atol: float = 0.0) -> bool:
    """True iff all pytree leaves of ``a`` and ``b`` are numerically equal."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    for la, lb in zip(leaves_a, leaves_b):
        if not np.allclose(np.asarray(la), np.asarray(lb), rtol=rtol, atol=atol):
            return False
    return True


def tree_l2_distance(a, b) -> float:
    """Sum of L2 norms of leaf-wise differences."""
    diffs = jax.tree_util.tree_map(lambda x, y: jnp.asarray(x - y), a, b)
    leaves = jax.tree_util.tree_leaves(diffs)
    if not leaves:
        return 0.0
    return float(
        sum(jnp.sqrt(jnp.sum(jnp.square(leaf))) for leaf in leaves)
    )


def build_teacher_config(
    paired_config: Mapping[str, Any],
    base_teacher_config: MutableMapping[str, Any],
    *,
    exclude_keys: Optional[set] = None,
) -> Dict[str, Any]:
    """Overlay shared knobs from the paired config onto a Quantized teacher cfg.

    Student-only keys (distill/BC/AR/diffusion/freeze) are skipped so they do
    not accidentally overwrite teacher ``distill_coef`` etc.
    """
    exclude = set(exclude_keys or ())
    exclude.update(
        {
            "agent_name",
            "freeze_teacher",
            "distill_coef",
            "token_distill_coef",
            "bc_coef",
            "dataset_bc_coef",
            "actor_emb_dim",
            "actor_depth",
            "actor_num_heads",
            "actor_dropout",
            "eval_sampling_temperature",
            "num_registers",
            "teacher_temperature",
            # Diffusion-student knobs (harmless if unused by AR).
            "diffusion_steps",
            "mask_ratio_min",
            "mask_ratio_max",
            "uniform_mix",
            "student_actor_hidden_dims",
            "soft_target_eps",
            "soft_target_temperature",
        }
    )
    teacher_cfg = dict(base_teacher_config)
    for key, value in paired_config.items():
        if key in exclude:
            continue
        if key in teacher_cfg or key in (
            "tokenizer_path",
            "projection_enabled",
            "h",
            "batch_size",
            "lr",
            "discount",
            "ensemble_ct",
            "flow_steps",
            "alpha",
            "expectile",
            "guidance_coef",
            "consensus_floor",
            "conflict_power",
            "residual_coef",
            "actor_hidden_dims",
            "value_hidden_dims",
            "guidance_hidden_dims",
        ):
            teacher_cfg[key] = value
    # Explicit teacher guidance-distill override if provided.
    if "teacher_distill_coef" in paired_config:
        teacher_cfg["distill_coef"] = paired_config["teacher_distill_coef"]
    teacher_cfg["agent_name"] = "quantized_dflrql9"
    return teacher_cfg


def finite_info(info: Mapping[str, Any]) -> bool:
    """True if every array-like info value is finite."""
    for value in info.values():
        arr = np.asarray(value)
        if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
            return False
    return True
