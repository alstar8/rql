"""QuantizedDFLRQL9: DFLRQL9 continuous flow with frozen OATTok deployment.

Continuous DFLRQL9 flow matching induces a pushforward token law
``π_Z(z|s) = P(Q(f_θ(s, ξ)) = z)`` via frozen OATTok encode/FSQ/decode.
The actor RL ``q_pe`` lookahead is scored after hard-forward / soft-backward
projection; critic / BC / distill paths stay on behavior actions (parent).
Deployment always returns the projected action chunk.

Training projection diagnostics (RMSE/MAE/token stats) are intentionally
not injected into ``total_loss`` (would require copying ~230 lines). Use
``scripts/eval_dflrql_oattok_projection.py`` or ``_project_flat_action`` /
``projection_diagnostics`` offline.

Paired-student helpers: ``sample_flow_actions_batch`` / ``sample_tokens_batch``
provide batched EMA (``target_actor`` + guidance) flow samples and frozen
OATTok token IDs with stop-gradient — call from Python orchestration, not
from inside another JIT.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from einops import rearrange

from agents.dflrql9 import DFLRQL9Agent, get_config as get_v9_config
from agents.oattok_jax import OATTok, load_tokenizer
from utils.flax_utils import nonpytree_field


class QuantizedDFLRQL9Agent(DFLRQL9Agent):
    """DFLRQL9 + frozen OATTok projection on actor q_pe and deploy only."""

    tokenizer_def: Any = nonpytree_field(default=None)
    tokenizer_params: Any = nonpytree_field(default=None)
    tokenizer_meta: Any = nonpytree_field(default=None)

    @staticmethod
    def assert_tokenizer_matches_policy(tok_meta, policy_h, prim_action_dim):
        sample_h = int(tok_meta["sample_horizon"])
        sample_dim = int(tok_meta["sample_dim"])
        if sample_h != int(policy_h):
            raise ValueError(
                f"Tokenizer sample_horizon={sample_h} does not match policy "
                f"h={policy_h}. Use the matching OATTok checkpoint "
                f"(e.g. humanoidmaze-large_h1_d21.pkl)."
            )
        if sample_dim != int(prim_action_dim):
            raise ValueError(
                f"Tokenizer sample_dim={sample_dim} does not match policy "
                f"primitive action dim={prim_action_dim}."
            )

    @staticmethod
    def build_oattok(tok_meta):
        return OATTok(
            sample_dim=int(tok_meta["sample_dim"]),
            sample_horizon=int(tok_meta["sample_horizon"]),
            num_registers=int(tok_meta["num_registers"]),
            emb_dim=int(tok_meta.get("emb_dim", 256)),
            encoder_depth=int(tok_meta.get("encoder_depth", 2)),
            decoder_depth=int(tok_meta.get("decoder_depth", 4)),
        )

    def _project_flat_action(self, flat_action):
        """Encode → FSQ (ST) → decode → clip; flat (B, h*Da) → same shape.

        Hard forward matches ``OATTok.__call__`` roundtrip; gradients flow to
        ``flat_action`` through frozen tokenizer params (never optimized).
        """
        if not self.config.get("projection_enabled", True):
            return flat_action
        h = int(self.config["h"])
        actions_btd = rearrange(flat_action, "b (h d) -> b h d", h=h)
        recons, _tokens, _quant = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            actions_btd,
            deterministic=True,
        )
        recons = jnp.clip(recons, -1.0, 1.0)
        return rearrange(recons, "b h d -> b (h d)")

    def _actor_q_action(self, flat_action):
        """Project actor q_pe lookahead so value scores quantized actions."""
        return self._project_flat_action(flat_action)

    def projection_diagnostics(self, flat_action):
        """Offline RMSE/MAE/saturation/token stats (not logged in total_loss)."""
        h = int(self.config["h"])
        actions_btd = rearrange(flat_action, "b (h d) -> b h d", h=h)
        recons, tokens, _quant = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            actions_btd,
            deterministic=True,
        )
        recons = jnp.clip(recons, -1.0, 1.0)
        delta = recons - actions_btd
        abs_delta = jnp.abs(delta)
        sat_eps = 1e-5
        return {
            "proj_rmse": jnp.sqrt(jnp.mean(jnp.square(delta))),
            "proj_mae": jnp.mean(abs_delta),
            "proj_max_abs": jnp.max(abs_delta),
            "raw_sat_frac": jnp.mean(jnp.abs(actions_btd) >= 1.0 - sat_eps),
            "proj_sat_frac": jnp.mean(jnp.abs(recons) >= 1.0 - sat_eps),
            "token_mean": tokens.astype(jnp.float32).mean(),
            "token_std": tokens.astype(jnp.float32).std(),
        }

    def _encode_flat_actions_to_tokens(self, flat_actions):
        """Frozen OATTok encode of flat ``(B, h*Da)`` → int tokens ``(B, K)``."""
        h = int(self.config["h"])
        actions_btd = rearrange(flat_actions, "b (h d) -> b h d", h=h)
        _quant, tokens = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            actions_btd,
            method=OATTok.encode,
            deterministic=True,
        )
        return tokens.astype(jnp.int32)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_flow_actions_batch(self, observations, seed, temperature=0.0):
        """Batched continuous flow via EMA actor+guidance (``temperature==0``).

        ``observations``: ``(B, obs_dim)``. Returns clipped flat actions
        ``(B, h*Da)``. Uses ``target_actor`` when ``temperature==0`` (same
        contract as ``compute_flow_actions``).
        """
        observations = jnp.atleast_2d(observations)
        batch_size = observations.shape[0]
        noise = jax.random.normal(
            seed,
            (batch_size, self.config["action_dim"]),
        )
        return self.compute_flow_actions(
            observations,
            noise=noise,
            seed=seed,
            temperature=temperature,
        )

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_tokens_batch(self, observations, seed, temperature=0.0):
        """EMA flow → frozen OATTok encode → stop-grad tokens ``(B, K)``.

        Intended for paired student distillation: call from Python (not from
        inside another jitted update). Token IDs are hard FSQ indices of the
        continuous flow sample; gradients into the teacher are stopped.

        Inlines batched flow (does not call ``sample_flow_actions_batch``) to
        avoid nested JIT.
        """
        observations = jnp.atleast_2d(observations)
        batch_size = observations.shape[0]
        flow_rng, _ = jax.random.split(seed)
        noise = jax.random.normal(
            flow_rng,
            (batch_size, self.config["action_dim"]),
        )
        flat_actions = self.compute_flow_actions(
            observations,
            noise=noise,
            seed=flow_rng,
            temperature=temperature,
        )
        tokens = self._encode_flat_actions_to_tokens(flat_actions)
        return jax.lax.stop_gradient(tokens)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        """Parent DFLRQL9 flow sample, then one OATTok project; shape (h, Da).

        Inlines the parent sample body (instead of calling the parent jitted
        ``sample_actions``) to avoid nested-JIT seed/trace mismatches, and
        projects only once after continuous flow.
        """
        action_rng, n_rng = jax.random.split(seed)
        obs = jnp.atleast_2d(obs)[-1:]
        noise = jax.random.normal(
            n_rng,
            (1, self.config["action_dim"]),
        )
        actions = self.compute_flow_actions(
            obs, seed=action_rng, noise=noise, temperature=temperature
        )
        projected = self._project_flat_action(actions)
        return rearrange(
            projected[0], "(h d) -> h d", h=self.config["h"]
        )

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Build exact DFLRQL9 network tree, then attach frozen OATTok."""
        config = dict(config)
        tokenizer_path = config.get("tokenizer_path", "")
        if not tokenizer_path:
            raise ValueError(
                "QuantizedDFLRQL9 requires agent.tokenizer_path to a frozen "
                "OATTok .pkl (e.g. exp/ogbench_oattok/humanoidmaze-large_h1_d21.pkl)."
            )

        tok_params, tok_meta = load_tokenizer(tokenizer_path)
        # Infer primitive action dim from example actions (same as parent create).
        ex_actions_arr = jnp.asarray(ex_actions)
        if ex_actions_arr.ndim >= 2:
            prim_dim = int(ex_actions_arr.shape[-1])
        else:
            prim_dim = int(ex_actions_arr.shape[0])
        cls.assert_tokenizer_matches_policy(
            tok_meta, config["h"], prim_dim
        )
        tokenizer_def = cls.build_oattok(tok_meta)

        # Prefer super.create so ModuleDict / optimizer match DFLRQL9 exactly.
        agent = super().create(seed, ex_observations, ex_actions, config)
        return agent.replace(
            tokenizer_def=tokenizer_def,
            tokenizer_params=tok_params,
            tokenizer_meta=tok_meta,
        )


def get_config():
    config = get_v9_config()
    config.agent_name = "quantized_dflrql9"
    # Required at create(); launcher / flags must set a real path.
    config.tokenizer_path = ""
    config.projection_enabled = True
    return config
