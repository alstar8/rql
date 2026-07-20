"""DiscreteCoordMaskIQL (v5): MaskGIT over FSQ coordinates + IQL/AWR.

Genuinely discrete successor to ConsensusDiscreteFlow (v3). Frozen OATTok
encode/decode; actor predicts per-axis mixed-radix coordinate logits under
Bernoulli masking; IQL Q/V with AWR-weighted masked CE. Deploy uses iterative
MaskGIT unmasking only — no critics at inference.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections as mlc
import numpy as np
import optax
from einops import rearrange

from agents.oattok_jax import (
    FSQ_DIM,
    FSQ_LEVELS,
    OATTok,
    coord_indices_to_token_ids,
    fsq_class_valid_mask,
    indices_to_codes,
    load_tokenizer,
    token_ids_to_coord_indices,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, Value, default_init

MAX_FSQ_CLASSES = int(max(FSQ_LEVELS))  # 8
FSQ_BASIS = (1, 8, 40, 200)  # mixed-radix basis for L=(8,5,5,5)


class CoordMaskActor(nn.Module):
    """MLP actor: (obs, flat coords, flat mask, t) -> (K, 4, 8) padded logits."""

    hidden_dims: Any
    num_registers: int
    num_axes: int = FSQ_DIM
    max_classes: int = MAX_FSQ_CLASSES
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs_coords_mask_t):
        h = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
        )(obs_coords_mask_t)
        logits = nn.Dense(
            self.num_registers * self.num_axes * self.max_classes,
            kernel_init=default_init(1.0),
        )(h)
        return logits.reshape(
            (-1, self.num_registers, self.num_axes, self.max_classes)
        )


class DiscreteCoordMaskIQLAgent(flax.struct.PyTreeNode):
    """Discrete coordinate-mask IQL agent for OGBench offline RL."""

    rng: Any
    network: Any
    tokenizer_def: Any = nonpytree_field()
    tokenizer_params: Any = nonpytree_field()
    class_valid: Any = nonpytree_field()
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(diff, expectile):
        """IQL expectile regression on residual ``target - prediction``."""
        weight = jnp.where(diff >= 0, expectile, (1.0 - expectile))
        return weight * (diff**2)

    @staticmethod
    def chunk_coord_ce_weights(rs_terminals, h):
        """Per-example CE weight: 0 if any of the h actions is post-terminal."""
        action_valid = 1.0 - rs_terminals[:h]
        return action_valid.min(axis=0)

    @staticmethod
    def safe_masked_mean(values, mask):
        """Mean of values over mask; 0 when mask has no True entries."""
        mask_f = mask.astype(values.dtype)
        denom = jnp.maximum(mask_f.sum(), 1.0)
        return (values * mask_f).sum() / denom

    @staticmethod
    def apply_class_valid_logits(logits, class_valid):
        """Set invalid class slots to a large negative value (never participate)."""
        # logits: (..., q, C); class_valid: (q, C)
        return jnp.where(class_valid, logits, jnp.asarray(-1e9, dtype=logits.dtype))

    @staticmethod
    def hard_coord_ce(logits, targets, class_valid, site_mask):
        """Hard CE over masked sites only; invalid classes excluded from softmax.

        logits: (B, K, q, C); targets: (B, K, q) int; site_mask: (B, K, q) bool/float.
        Returns per-site CE (B, K, q) (undefined sites zeroed by mask in callers).
        """
        safe_logits = DiscreteCoordMaskIQLAgent.apply_class_valid_logits(
            logits, class_valid
        )
        log_probs = jax.nn.log_softmax(safe_logits, axis=-1)
        oh = jax.nn.one_hot(targets, safe_logits.shape[-1], dtype=log_probs.dtype)
        # Drop any mass on invalid slots (targets are always in-range).
        oh = oh * class_valid.astype(oh.dtype)
        ce = -(oh * log_probs).sum(axis=-1)
        return ce * site_mask.astype(ce.dtype)

    @staticmethod
    def corrupt_coords(rng, clean_coords, t):
        """Bernoulli mask with p=1-t; guarantee ≥1 masked site per example.

        Returns (masked_coords, mask, input_coords) where mask True = masked,
        input_coords has zeros on masked sites (no identity-copy of those values).
        """
        # clean_coords: (B, K, num_axes); t: (B, 1)
        b, k, num_axes = clean_coords.shape
        m = k * num_axes
        rng, m_rng, f_rng = jax.random.split(rng, 3)
        keep_prob = t  # clean fraction
        # mask True => corrupted / masked for CE
        mask_flat = jax.random.uniform(m_rng, (b, m)) >= keep_prob
        force_idx = jax.random.randint(f_rng, (b,), 0, m)
        force = jax.nn.one_hot(force_idx, m, dtype=bool)
        any_masked = mask_flat.any(axis=-1, keepdims=True)
        mask_flat = jnp.where(any_masked, mask_flat, force)
        mask = mask_flat.reshape(b, k, num_axes)
        input_coords = jnp.where(mask, 0, clean_coords).astype(jnp.float32)
        return clean_coords, mask, input_coords

    @staticmethod
    def awr_ramp_fraction(step, bc_warmup_steps, awr_ramp_steps):
        """0 during BC warmup, then linear ramp to 1 over awr_ramp_steps."""
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warmup = jnp.asarray(bc_warmup_steps, dtype=jnp.float32)
        ramp = jnp.maximum(jnp.asarray(awr_ramp_steps, dtype=jnp.float32), 1.0)
        return jnp.clip(jnp.maximum(step_f - warmup, 0.0) / ramp, 0.0, 1.0)

    @staticmethod
    def awr_example_weights(advantage, temperature, max_weight, ramp_frac):
        """Blend identity BC weights into clipped exp(A/T) via ramp_frac."""
        awr = jnp.exp(advantage / jnp.maximum(temperature, 1e-6))
        awr = jnp.minimum(awr, max_weight)
        ones = jnp.ones_like(awr)
        return (1.0 - ramp_frac) * ones + ramp_frac * awr

    @staticmethod
    def awr_ess(weights):
        """Effective sample size of positive weights; 0 if empty."""
        w = jnp.maximum(weights, 0.0)
        denom = jnp.maximum(jnp.square(w).sum(), 1e-12)
        return jnp.square(w.sum()) / denom

    @staticmethod
    def h_step_td_target(rewards, terminals, masks, discount, discount_mul, h, next_v):
        """RQL h-step TD target with right-shifted terminals and V bootstrap.

        rewards/terminals/masks: (H+, B); next_v: (B,); returns (target, valids).
        """
        rs_terminals = jnp.concatenate(
            [jnp.zeros_like(terminals[:1]), terminals[:-1]],
            axis=0,
        )
        n_rews = (rewards * discount_mul[..., None] * (1.0 - rs_terminals)).sum(0)
        target = n_rews + (discount**h) * next_v * masks[-2]
        terminal_count = rs_terminals.sum(0)
        valids = (terminal_count <= 1).astype(target.dtype)
        return target, valids, rs_terminals

    @staticmethod
    def maskgit_remaining_counts(num_coords, n_steps):
        """Cosine remaining-mask schedule; final step forces 0 remaining.

        Every non-final step strictly decreases the remaining count whenever
        anything is still masked (avoids the M=64,N=16 step-0 64→64 no-op).
        """
        counts = []
        for i in range(n_steps):
            if i == n_steps - 1:
                counts.append(0)
            else:
                frac = float(np.cos(0.5 * np.pi * (i + 1) / n_steps))
                counts.append(int(np.ceil(frac * num_coords)))
        # Ensure nonincreasing, bounded, and strict progress on non-final steps.
        out = []
        prev = num_coords
        for i, c in enumerate(counts):
            c = int(min(max(c, 0), prev))
            if i < n_steps - 1 and prev > 0:
                c = min(c, prev - 1)
            out.append(c)
            prev = c
        out[-1] = 0
        return tuple(out)

    @staticmethod
    def unmask_topk_by_confidence(mask, confidence, num_to_unmask):
        """Unmask the highest-confidence still-masked sites.

        mask True = still masked. ``num_to_unmask`` may be a scalar or ``(B,)``.
        Fixed-shape JIT-safe via argsort ranking.
        """
        # mask, confidence: (B, M)
        scores = jnp.where(mask, confidence, jnp.asarray(-jnp.inf, dtype=confidence.dtype))
        order = jnp.argsort(-scores, axis=-1)
        ranks = jnp.argsort(order, axis=-1)  # 0 = highest confidence
        n = jnp.asarray(num_to_unmask).reshape(-1)
        # Broadcast (1,) or (B,) against ranks (B, M).
        unmask = (ranks < n[:, None]) & mask
        return mask & (~unmask)

    def _encode_actions(self, actions_btd):
        """(B, T, D) -> tokens (B, K), codes (B, K, q)."""
        quant, tokens = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            actions_btd,
            method=OATTok.encode,
            deterministic=True,
        )
        return tokens, quant

    def _decode_codes(self, codes):
        """(B, K, q) -> flattened continuous action (B, h*Da)."""
        recons = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            codes,
            method=OATTok.decode,
            deterministic=True,
        )
        return rearrange(recons, "b t d -> b (t d)")

    def _actor_input(self, observations, input_coords, mask, times):
        """Build actor features: obs || flat coords || flat mask || t."""
        flat_coords = rearrange(input_coords, "b k q -> b (k q)")
        flat_mask = rearrange(mask.astype(jnp.float32), "b k q -> b (k q)")
        return jnp.concatenate([observations, flat_coords, flat_mask, times], axis=-1)

    def _coord_logits(self, observations, input_coords, mask, times, actor_name, params=None):
        logits = self.network.select(actor_name)(
            self._actor_input(observations, input_coords, mask, times),
            params=params,
        )
        return self.apply_class_valid_logits(logits, self.class_valid)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        batch_size = self.config["batch_size"]
        h = self.config["h"]
        num_axes = FSQ_DIM

        rng, t_rng, c_rng = jax.random.split(rng, 3)

        # ---- Encode clean action chunks -> discrete coords ----
        actions_hbd = batch["actions"][:h]
        actions_btd = rearrange(actions_hbd, "h b d -> b h d")
        clean_tokens, clean_codes = self._encode_actions(actions_btd)
        clean_tokens = jax.lax.stop_gradient(clean_tokens)
        clean_codes = jax.lax.stop_gradient(clean_codes)
        clean_coords = token_ids_to_coord_indices(clean_tokens, FSQ_LEVELS)
        clean_coords = jax.lax.stop_gradient(clean_coords)
        clean_actions = jax.lax.stop_gradient(self._decode_codes(clean_codes))

        # ---- IQL Q on clean decoded actions (no t) ----
        q_in = jnp.concatenate([batch["observations"][0], clean_actions], axis=-1)
        q_values = self.network.select("q")(q_in, params=grad_params)

        next_v = self.network.select("target_v")(batch["observations"][-1])
        if next_v.ndim > 1:
            next_v = next_v.mean(axis=0)
        target_q, valids, rs_terminals = self.h_step_td_target(
            batch["rewards"],
            batch["terminals"],
            batch["masks"],
            self.config["discount"],
            self.config["discount_mul"],
            h,
            next_v,
        )
        # MSE to stop-grad TD target (IQL Q).
        td_target = jax.lax.stop_gradient(target_q)
        q_err = q_values - td_target  # (ensemble, B) or (B,)
        if q_err.ndim == 1:
            q_loss = (jnp.square(q_err) * valids).mean()
            q_mean = q_values.mean()
        else:
            q_loss = (jnp.square(q_err) * valids[None, :]).mean()
            q_mean = q_values.mean()

        # ---- V expectile to stop-grad target_q(s, a_behavior) ----
        target_q_values = self.network.select("target_q")(q_in)
        if target_q_values.ndim > 1:
            q_behavior = target_q_values.mean(axis=0)
            if self.config["rho"] != 0.0:
                q_behavior = q_behavior - self.config["rho"] * target_q_values.std(
                    axis=0
                )
        else:
            q_behavior = target_q_values
        q_behavior = jax.lax.stop_gradient(q_behavior)

        v_values = self.network.select("v")(
            batch["observations"][0], params=grad_params
        )
        if v_values.ndim > 1:
            v_pred = v_values.mean(axis=0)
            v_diff = q_behavior - v_values
            v_loss = (
                self.expectile_loss(v_diff, self.config["expectile"]).mean(axis=0)
                * valids
            ).mean()
        else:
            v_pred = v_values
            v_diff = q_behavior - v_values
            v_loss = (
                self.expectile_loss(v_diff, self.config["expectile"]) * valids
            ).mean()

        # ---- Advantage + AWR weights (raw temperature; no A standardization) ----
        advantage = jax.lax.stop_gradient(q_behavior - v_pred)
        ramp_frac = self.awr_ramp_fraction(
            self.network.step,
            self.config["bc_warmup_steps"],
            self.config["awr_ramp_steps"],
        )
        awr_w = self.awr_example_weights(
            advantage,
            self.config["awr_temperature"],
            self.config["max_weight"],
            ramp_frac,
        )
        chunk_w = self.chunk_coord_ce_weights(rs_terminals, h)
        example_w = awr_w * chunk_w * valids

        # ---- Masked coordinate CE (hard targets) ----
        # Sample Bernoulli keep-prob, then condition the actor on the *realized*
        # clean fraction so train t matches inference (actual unmasked share).
        t_keep = jax.random.uniform(t_rng, (batch_size, 1))
        t_keep = jnp.clip(t_keep, 1e-3, 1.0 - 1e-3)
        _, mask, input_coords = self.corrupt_coords(c_rng, clean_coords, t_keep)
        t = (
            1.0 - mask.astype(jnp.float32).mean(axis=(1, 2))
        ).reshape(batch_size, 1)
        logits = self.network.select("actor")(
            self._actor_input(batch["observations"][0], input_coords, mask, t),
            params=grad_params,
        )
        ce_site = self.hard_coord_ce(logits, clean_coords, self.class_valid, mask)
        # Per-example mean CE over masked sites, then weight-normalize.
        mask_f = mask.astype(ce_site.dtype)
        masked_count = jnp.maximum(mask_f.sum(axis=(1, 2)), 1.0)
        ce_ex = ce_site.sum(axis=(1, 2)) / masked_count
        weight_sum = jnp.maximum(example_w.sum(), 1e-6)
        actor_loss = (ce_ex * example_w).sum() / weight_sum

        total_loss = (
            self.config["q_coef"] * q_loss
            + self.config["v_coef"] * v_loss
            + self.config["alpha"] * actor_loss
        )

        # ---- Diagnostics ----
        safe_logits = self.apply_class_valid_logits(logits, self.class_valid)
        probs = jax.nn.softmax(safe_logits, axis=-1)
        pred = jnp.argmax(safe_logits, axis=-1)
        correct = (pred == clean_coords).astype(jnp.float32)
        levels = jnp.asarray(FSQ_LEVELS)
        axis_acc = []
        axis_ce = []
        for axi in range(num_axes):
            m_ax = mask[:, :, axi]
            axis_acc.append(self.safe_masked_mean(correct[:, :, axi], m_ax))
            axis_ce.append(self.safe_masked_mean(ce_site[:, :, axi], m_ax))

        actions_flat = rearrange(actions_btd, "b h d -> b (h d)")
        decode_rmse = jnp.sqrt(jnp.mean(jnp.square(clean_actions - actions_flat)))

        clip_frac = (awr_w >= self.config["max_weight"] - 1e-6).astype(jnp.float32).mean()
        ess = self.awr_ess(example_w)

        # Grad metrics: norms of module params under this loss (via stop-grad leaves).
        # Reported later from update; here store loss-scale proxies.
        info = {
            "total_loss": total_loss,
            "q_loss": q_loss,
            "v_loss": v_loss,
            "actor_loss": actor_loss,
            "coord_acc": correct.mean(),
            "coord_acc_masked": self.safe_masked_mean(correct, mask),
            "mask_fraction": mask_f.mean(),
            "ce_masked": self.safe_masked_mean(ce_site, mask),
            "q": q_mean,
            "q_mean": q_mean,
            "q_max": jnp.max(q_values),
            "q_min": jnp.min(q_values),
            "v_mean": v_pred.mean(),
            "v_max": v_pred.max(),
            "v_min": v_pred.min(),
            "adv_mean": advantage.mean(),
            "adv_std": advantage.std(),
            "adv_max": advantage.max(),
            "adv_min": advantage.min(),
            "awr_weight_mean": awr_w.mean(),
            "awr_weight_max": awr_w.max(),
            "awr_clip_frac": clip_frac,
            "awr_ess": ess,
            "actor_weight_ramp": ramp_frac,
            "example_weight_mean": example_w.mean(),
            "chunk_weight_mean": chunk_w.mean(),
            "decode_rmse": decode_rmse,
            "t_mean": t.mean(),
            "valids_mean": valids.mean(),
        }
        for axi in range(num_axes):
            info[f"axis{axi}_acc_masked"] = axis_acc[axi]
            info[f"axis{axi}_ce_masked"] = axis_ce[axi]
            info[f"axis{axi}_levels"] = levels[axi].astype(jnp.float32)
        # Unused but keeps probs referenced for XLA friendliness in diagnostics.
        info["pred_prob_mean"] = self.safe_masked_mean(probs.max(axis=-1), mask)
        return total_loss, info

    def target_update(self, network, module_name, d):
        """Polyak/EMA from *post-update* online params."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        # Grad norms from the just-computed update (TrainState may stash grads;
        # fall back to finite loss proxies already in info).
        def _leaf_norm(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            if not leaves:
                return jnp.zeros(())
            return jnp.sqrt(sum(jnp.square(x).sum() for x in leaves))

        # Approximate module grad presence via parameter delta magnitude.
        for name in ("q", "v", "actor"):
            delta = jax.tree_util.tree_map(
                lambda a, b: a - b,
                new_network.params[f"modules_{name}"],
                self.network.params[f"modules_{name}"],
            )
            info[f"grad_delta_{name}"] = _leaf_norm(delta)

        self.target_update(new_network, "q", d=self.config["tau"])
        self.target_update(new_network, "v", d=self.config["tau"])
        self.target_update(new_network, "actor", d=1.0 - self.config["ema"])
        return self.replace(network=new_network, rng=new_rng), info

    def _sample_or_argmax(self, logits, rng, temperature):
        """Return (coords, confidence, new_rng). confidence in [0,1] per site."""
        # logits: (B, K, q, C) already class-masked
        probs = jax.nn.softmax(logits, axis=-1)
        if temperature == 0.0:
            coords = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            confidence = probs.max(axis=-1)
            return coords, confidence, rng

        rng, s_rng, g_rng = jax.random.split(rng, 3)
        # Optional Gumbel-adjusted confidence for sampling path.
        gumbel = -jnp.log(
            -jnp.log(jax.random.uniform(g_rng, logits.shape, dtype=logits.dtype) + 1e-8)
            + 1e-8
        )
        sample_logits = logits / jnp.maximum(temperature, 1e-6)
        flat = rearrange(sample_logits, "b k q c -> (b k q) c")
        flat_coords = jax.random.categorical(s_rng, flat).astype(jnp.int32)
        coords = flat_coords.reshape(logits.shape[:3])
        # Confidence: Gumbel-perturbed softmax prob of chosen class.
        pert_probs = jax.nn.softmax(sample_logits + gumbel, axis=-1)
        confidence = jnp.take_along_axis(
            pert_probs, coords[..., None], axis=-1
        ).squeeze(-1)
        return coords, confidence, rng

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_maskgit_actions(self, observations, seed=None, temperature=0.0):
        """Iterative MaskGIT over K*q coordinates; decode via frozen OATTok.

        Starts fully masked. Uses target_actor when temperature==0 else online
        actor. Cosine remaining-mask schedule controls *how many* sites to
        unmask each step; the actor time channel is the realized clean
        fraction ``1 - mask.mean``, matching training. Final step unmasks
        everything. No critic at deploy.

        Seed contract (main/eval always pass a seed via ``supply_rng``):
        - ``temperature==0``: deterministic argmax; seed is unused after the
          all-mask start.
        - ``temperature>0`` with a supplied seed: fully reproducible.
        - ``seed is None``: falls back to ``self.rng`` but does **not** write
          the advanced key back — pure repeated calls with ``seed=None``
          therefore repeat the same sample (no pretended state mutation).
        """
        b = observations.shape[0]
        k = self.config["num_registers"]
        num_axes = FSQ_DIM
        m = k * num_axes
        n = int(self.config["maskgit_steps"])
        remaining_schedule = self.maskgit_remaining_counts(m, n)

        coords = jnp.zeros((b, k, num_axes), dtype=jnp.int32)
        mask = jnp.ones((b, k, num_axes), dtype=bool)
        actor_name = "actor" if temperature > 0.0 else "target_actor"
        rng = seed if seed is not None else self.rng

        for i in range(n):
            rng, step_rng = jax.random.split(rng)
            # Realized clean fraction (matches training actor conditioning).
            t = (
                1.0 - mask.astype(jnp.float32).mean(axis=(1, 2))
            ).reshape(b, 1)
            input_coords = jnp.where(mask, 0, coords).astype(jnp.float32)
            logits = self._coord_logits(
                observations, input_coords, mask, t, actor_name
            )
            cand, confidence, _ = self._sample_or_argmax(
                logits, step_rng, temperature
            )
            # Write predictions only into still-masked sites.
            coords = jnp.where(mask, cand, coords)
            remain = remaining_schedule[i]
            mask_flat = mask.reshape(b, m)
            conf_flat = confidence.reshape(b, m)
            currently = mask_flat.sum(axis=-1)
            # Unmask highest-confidence sites until `remain` are still masked.
            to_unmask = jnp.maximum(currently - remain, 0)
            if i == n - 1:
                mask = jnp.zeros((b, k, num_axes), dtype=bool)
            else:
                mask = self.unmask_topk_by_confidence(
                    mask_flat, conf_flat, to_unmask
                ).reshape(b, k, num_axes)

        tokens = coord_indices_to_token_ids(coords, FSQ_LEVELS)
        codes = indices_to_codes(tokens, FSQ_LEVELS)
        actions = self._decode_codes(codes)
        return jnp.clip(actions, -1.0, 1.0)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        """Sample an action chunk; see ``compute_maskgit_actions`` seed contract."""
        obs = jnp.atleast_2d(obs)[-1:]
        actions = self.compute_maskgit_actions(
            obs, seed=seed, temperature=temperature
        )[0]
        actions = rearrange(actions, "(h d) -> h d", h=self.config["h"])
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_actions = jnp.asarray(ex_actions)
        ex_observations = jnp.asarray(ex_observations)
        if ex_actions.ndim == 3:
            prim_dim = int(ex_actions.shape[-1])
            ex_obs = ex_observations[0]
            ex_act_step = ex_actions[0]
        elif ex_actions.ndim == 2:
            prim_dim = int(ex_actions.shape[-1])
            ex_obs = (
                ex_observations if ex_observations.ndim == 2 else ex_observations[None, :]
            )
            ex_act_step = ex_actions
        else:
            prim_dim = int(ex_actions.shape[0])
            ex_obs = (
                ex_observations[None, :]
                if ex_observations.ndim == 1
                else ex_observations
            )
            ex_act_step = ex_actions[None, :]

        h = int(config["h"])
        action_dim = prim_dim * h
        num_registers = int(config["num_registers"])

        tokenizer_path = config.get("tokenizer_path", "")
        if not tokenizer_path:
            raise ValueError(
                "DiscreteCoordMaskIQL requires agent.tokenizer_path to a "
                "frozen OATTok checkpoint (.pkl)."
            )
        tok_params, tok_meta = load_tokenizer(tokenizer_path)
        if int(tok_meta["sample_dim"]) != prim_dim:
            raise ValueError(
                f"Tokenizer sample_dim={tok_meta['sample_dim']} != action dim {prim_dim}"
            )
        if int(tok_meta["sample_horizon"]) != h:
            raise ValueError(
                f"Tokenizer sample_horizon={tok_meta['sample_horizon']} != h={h}"
            )
        num_registers = int(tok_meta.get("num_registers", num_registers))
        config = dict(config)
        config["num_registers"] = num_registers

        tokenizer_def = OATTok(
            sample_dim=prim_dim,
            sample_horizon=h,
            num_registers=num_registers,
            emb_dim=int(tok_meta.get("emb_dim", 256)),
            encoder_depth=int(tok_meta.get("encoder_depth", 2)),
            decoder_depth=int(tok_meta.get("decoder_depth", 4)),
        )

        ex_flat_actions = jnp.concatenate([ex_act_step] * h, axis=-1)
        ex_times = jnp.zeros((ex_obs.shape[0], 1), dtype=jnp.float32)
        ex_coords = jnp.zeros(
            (ex_obs.shape[0], num_registers, FSQ_DIM), dtype=jnp.float32
        )
        ex_mask = jnp.ones_like(ex_coords)
        ex_actor_in = jnp.concatenate(
            [
                ex_obs,
                rearrange(ex_coords, "b k q -> b (k q)"),
                rearrange(ex_mask, "b k q -> b (k q)"),
                ex_times,
            ],
            axis=-1,
        )
        ex_q_in = jnp.concatenate([ex_obs, ex_flat_actions], axis=-1)

        q_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )
        v_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["v_ensemble_ct"],
        )
        actor_def = CoordMaskActor(
            hidden_dims=config["actor_hidden_dims"],
            num_registers=num_registers,
            layer_norm=config["layer_norm"],
        )

        network_info = dict(
            q=(q_def, (ex_q_in,)),
            target_q=(copy.deepcopy(q_def), (ex_q_in,)),
            v=(v_def, (ex_obs,)),
            target_v=(copy.deepcopy(v_def), (ex_obs,)),
            actor=(actor_def, (ex_actor_in,)),
            target_actor=(copy.deepcopy(actor_def), (ex_actor_in,)),
        )
        networks = {name: spec[0] for name, spec in network_info.items()}
        network_args = {name: spec[1] for name, spec in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_q"] = params["modules_q"]
        params["modules_target_v"] = params["modules_v"]
        params["modules_target_actor"] = params["modules_actor"]

        config["action_dim"] = action_dim
        config["prim_action_dim"] = prim_dim
        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(h)) + [jnp.inf])
        )
        class_valid = fsq_class_valid_mask(FSQ_LEVELS, MAX_FSQ_CLASSES)

        return cls(
            rng=rng,
            network=network,
            tokenizer_def=tokenizer_def,
            tokenizer_params=tok_params,
            class_valid=class_valid,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="discrete_coord_mask_iql",
            h=1,
            alpha=1.0,
            q_coef=1.0,
            v_coef=1.0,
            expectile=0.7,
            ensemble_ct=2,
            v_ensemble_ct=1,
            rho=0.0,
            lr=3e-4,
            discount=0.995,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            tau=0.005,
            ema=0.999,
            maskgit_steps=16,
            num_registers=16,
            awr_temperature=40.0,
            max_weight=100.0,
            bc_warmup_steps=50_000,
            awr_ramp_steps=50_000,
            # Advantages are raw (not standardized); temperature 40 from dataset
            # advantage scale evidence so exp(A/T) is not immediately saturated.
            tokenizer_path="",
        )
    )
    return config
