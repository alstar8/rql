"""ConsensusLatentFlow (v4): continuous flow matching in frozen FSQ latent space.

Evidence from ConsensusDiscreteFlow (v3): the OATTok tokenizer reconstructs
well, but categorical posterior denoising + logit-tilt guidance failed and Q
collapsed. This agent keeps the frozen tokenizer and continuous critics on
decoded actions, but replaces categorical token dynamics with continuous
vector-field flow matching inside the FSQ latent box.
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
import optax
from einops import rearrange

from agents.oattok_jax import (
    CODEBOOK_SIZE,
    FSQ_DIM,
    FSQ_LEVELS,
    OATTok,
    build_codebook,
    indices_to_codes,
    load_tokenizer,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, Value


class LatentFlowActor(nn.Module):
    """Continuous velocity MLP: (s, flat latent, t) -> (K*q)."""

    hidden_dims: Any
    latent_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs_latent_times):
        return MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(obs_latent_times)


class LatentGuidance(nn.Module):
    """Guidance head over flattened FSQ latents: (s, latent, t) -> (K*q)."""

    hidden_dims: Any
    latent_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs_latent_times):
        return MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(obs_latent_times)


class ConsensusLatentFlowAgent(flax.struct.PyTreeNode):
    """ConsensusLatentFlow agent for OGBench offline RL."""

    rng: Any
    network: Any
    tokenizer_def: Any = nonpytree_field()
    tokenizer_params: Any = nonpytree_field()
    codebook: Any = nonpytree_field()
    latent_box_min: Any = nonpytree_field()
    latent_box_max: Any = nonpytree_field()
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @staticmethod
    def codebook_box_bounds(codebook, num_registers):
        """Per-dim codebook min/max tiled over K registers -> flat (K*q,)."""
        # codebook: (V, q)
        lo = codebook.min(axis=0)
        hi = codebook.max(axis=0)
        lo_flat = jnp.tile(lo, num_registers)
        hi_flat = jnp.tile(hi, num_registers)
        return lo_flat, hi_flat

    @staticmethod
    def project_box(x, box_min, box_max):
        """Clip flattened latents into the FSQ codebook AABB."""
        return jnp.clip(x, box_min, box_max)

    @staticmethod
    def sample_uniform_token_latent(rng, batch_size, num_registers):
        """Sample uniform token ids and convert to flattened FSQ codes."""
        tokens = jax.random.randint(rng, (batch_size, num_registers), 0, CODEBOOK_SIZE)
        codes = indices_to_codes(tokens, FSQ_LEVELS)
        return rearrange(codes, "b k q -> b (k q)"), tokens

    @staticmethod
    def flow_interpolant(x0, x1, t):
        """Linear flow matching interpolant and target velocity."""
        x_t = (1.0 - t) * x0 + t * x1
        target_velocity = x1 - x0
        return x_t, target_velocity

    @staticmethod
    def actor_weight_from_step(step, bc_warmup_steps, actor_ramp_steps, actor_coef):
        """Piecewise actor RL weight: 0 during BC warmup, then linear ramp."""
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warmup = jnp.asarray(bc_warmup_steps, dtype=jnp.float32)
        ramp = jnp.maximum(jnp.asarray(actor_ramp_steps, dtype=jnp.float32), 1.0)
        coef = jnp.asarray(actor_coef, dtype=jnp.float32)
        after_warmup = jnp.maximum(step_f - warmup, 0.0)
        frac = jnp.clip(after_warmup / ramp, 0.0, 1.0)
        return coef * frac

    @staticmethod
    def chunk_bc_weights(rs_terminals, h):
        """Per-example BC weight: 0 if any of the h actions is post-terminal."""
        action_valid = 1.0 - rs_terminals[:h]
        return action_valid.min(axis=0)

    @staticmethod
    def final_action_shape(batch_size, h, prim_dim):
        """Helper: flattened decoded action shape (B, h*Da)."""
        return (batch_size, h * prim_dim)

    def _project_unit_ball(self, w):
        w_norm = jnp.linalg.norm(w, axis=-1, keepdims=True)
        return w * jnp.minimum(1.0, 1.0 / (w_norm + 1e-6))

    def _behavior_safe_direction(self, w, behavior_velocity):
        """Trust-weighted conflict + residual damping (dflrql9 / ConsensusFlow)."""
        w = self._project_unit_ball(w)
        w_norm = jnp.linalg.norm(w, axis=-1, keepdims=True)
        trust = jax.lax.stop_gradient(w_norm)

        behavior_velocity = jax.lax.stop_gradient(behavior_velocity)
        velocity_norm = jnp.linalg.norm(behavior_velocity, axis=-1, keepdims=True)
        velocity_unit = behavior_velocity / jnp.maximum(velocity_norm, 1e-6)
        parallel = (w * velocity_unit).sum(axis=-1, keepdims=True)

        power = self.config["conflict_power"]
        kill_frac = 1.0 - jnp.power(jnp.clip(trust, 0.0, 1.0), power)
        conflicting_parallel = jnp.minimum(parallel, 0.0)
        conflict_free = w - kill_frac * conflicting_parallel * velocity_unit
        conflict_free = self._project_unit_ball(conflict_free)

        alignment_cos = parallel / (w_norm + 1e-6)
        residual_coef = self.config["residual_coef"]
        damp = 1.0 - residual_coef * jnp.maximum(alignment_cos, 0.0) * trust
        damp = jnp.clip(damp, 0.0, 1.0)
        safe_w = self._project_unit_ball(conflict_free * damp)
        diagnostics = {
            "behavior_alignment_cos": alignment_cos,
            "behavior_conflict": (parallel < 0.0).astype(w.dtype),
            "conflict_kill_frac": kill_frac,
            "guidance_retained": jnp.linalg.norm(safe_w, axis=-1, keepdims=True)
            / (w_norm + 1e-6),
            "residual_damp": damp,
            "safe_w_norm": jnp.linalg.norm(safe_w, axis=-1, keepdims=True),
            "trust": trust,
        }
        return safe_w, diagnostics

    def _encode_actions(self, actions_btd):
        """(B, T, D) -> tokens (B, K), codes (B, K, q)."""
        quant, tokens = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            actions_btd,
            method=OATTok.encode,
            deterministic=True,
        )
        return tokens, quant

    def _decode_flat_latent(self, flat_latent):
        """(B, K*q) -> flattened continuous action (B, h*Da)."""
        codes = rearrange(
            flat_latent,
            "b (k q) -> b k q",
            k=self.config["num_registers"],
            q=FSQ_DIM,
        )
        recons = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            codes,
            method=OATTok.decode,
            deterministic=True,
        )
        return rearrange(recons, "b t d -> b (t d)")

    def _actor_input(self, observations, flat_latent, times):
        return jnp.concatenate([observations, flat_latent, times], axis=-1)

    def guidance_field(self, observations, flat_latent, times, behavior_velocity, stop_guidance=False):
        """G = lambda * t * u_safe in flat latent space."""
        w = self.network.select("guidance")(
            self._actor_input(observations, flat_latent, times)
        )
        safe_w, safety = self._behavior_safe_direction(w, behavior_velocity)
        if stop_guidance:
            safe_w = jax.lax.stop_gradient(safe_w)
        g = self.config["guidance_coef"] * times * safe_w
        return g, w, safe_w, safety

    def _reverse_guided_latent(self, observations, x1, d_b):
        """Reverse the guided latent ODE from clean x1,t=1 for random duration."""
        x_f = jnp.copy(x1)
        f = jnp.ones((x1.shape[0], 1), dtype=x1.dtype)
        n = int(self.config["flow_steps"])
        for _ in range(n):
            fm = self._actor_input(observations, x_f, f)
            behavior_velocity = self.network.select("actor")(fm)
            guidance, _, _, _ = self.guidance_field(
                observations, x_f, f, behavior_velocity, stop_guidance=False
            )
            x_f = self.project_box(
                x_f - (behavior_velocity + guidance) * d_b[..., None],
                self.latent_box_min,
                self.latent_box_max,
            )
            f = f - d_b[..., None]
        return x_f, f

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        batch_size = self.config["batch_size"]
        h = self.config["h"]
        num_registers = self.config["num_registers"]
        n_steps = self.config["flow_steps"]

        # ---- Critic bootstrap at next state: uniform-token latent decode ----
        rng, n_rng, u_rng, r_rng, t_rng, k_rng, a_rng = jax.random.split(rng, 7)
        noise_latent, _ = self.sample_uniform_token_latent(
            n_rng, batch_size, num_registers
        )
        noise_actions = jax.lax.stop_gradient(self._decode_flat_latent(noise_latent))
        next_state = jnp.concatenate(
            [
                batch["observations"][-1],
                noise_actions,
                jnp.zeros((batch_size, 1)),
            ],
            axis=-1,
        )
        next_qs = self.network.select("target_value")(next_state)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        # ---- Encode clean action chunks -> stop-grad quantized codes ----
        actions_hbd = batch["actions"][:h]
        actions_btd = rearrange(actions_hbd, "h b d -> b h d")
        clean_tokens, clean_codes = self._encode_actions(actions_btd)
        clean_tokens = jax.lax.stop_gradient(clean_tokens)
        clean_codes = jax.lax.stop_gradient(clean_codes)
        x1 = rearrange(clean_codes, "b k q -> b (k q)")

        # ---- Critic states: reverse guided latent ODE (ConsensusFlow-style) ----
        d = jnp.concatenate(
            [
                jax.random.uniform(u_rng, (batch_size // 2,)),
                jax.random.randint(r_rng, (batch_size // 2,), 0, n_steps + 1)
                / n_steps,
            ],
            axis=0,
        )
        d_b = d / n_steps
        x_f, f = self._reverse_guided_latent(batch["observations"][0], x1, d_b)
        y_f = jax.lax.stop_gradient(self._decode_flat_latent(x_f))
        state = jnp.concatenate(
            [batch["observations"][0], y_f, jax.lax.stop_gradient(f)],
            axis=-1,
        )
        q = self.network.select("value")(state, params=grad_params)

        rs_terminals = jnp.concatenate(
            [jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]],
            axis=0,
        )
        n_rews = (
            batch["rewards"]
            * self.config["discount_mul"][..., None]
            * (1 - rs_terminals)
        ).sum(0)
        target_q = (
            n_rews
            + (self.config["discount"] ** h) * next_q * batch["masks"][-2]
        )
        terminal_count = rs_terminals.sum(0)
        valids = (terminal_count <= 1).astype(terminal_count.dtype)
        critic_loss = (
            self.expectile_loss(target_q - q, target_q - q, self.config["expectile"])
            * valids
        ).mean()

        # ---- Latent flow-matching BC ----
        x0, _ = self.sample_uniform_token_latent(a_rng, batch_size, num_registers)
        t = jax.random.uniform(t_rng, (batch_size, 1))
        t = jnp.clip(t, 1e-3, 1.0 - 1e-3)
        x_t, target_velocity = self.flow_interpolant(x0, x1, t)
        fm_in = self._actor_input(batch["observations"][0], x_t, t)
        behavior_velocity = self.network.select("actor")(fm_in, params=grad_params)
        bc_weight = self.chunk_bc_weights(rs_terminals, h)
        bc_per = jnp.square(behavior_velocity - target_velocity).mean(axis=-1)
        bc_loss = (bc_per * bc_weight).sum() / jnp.maximum(bc_weight.sum(), 1e-6)

        # ---- Guidance distillation (scale-free consensus through decoder) ----
        member = jax.random.randint(k_rng, (batch_size,), 0, self.config["ensemble_ct"])
        member_onehot = jax.nn.one_hot(member, self.config["ensemble_ct"])

        def q_member_sum(flat_codes):
            y = self._decode_flat_latent(flat_codes)
            q_in = jnp.concatenate([batch["observations"][0], y, t], axis=-1)
            qs = self.network.select("target_value")(q_in)
            return (qs * member_onehot.T).sum()

        q_grad = jax.lax.stop_gradient(jax.grad(q_member_sum)(x_t))
        q_grad_norm = jnp.linalg.norm(q_grad, axis=-1, keepdims=True)
        valid_count = valids.sum() + 1e-6
        grad_scale = jax.lax.stop_gradient(
            (q_grad_norm[..., 0] * valids).sum() / valid_count
        )
        relative_floor = self.config["consensus_floor"] * grad_scale
        consensus_target = q_grad / (q_grad_norm + relative_floor + 1e-6)

        w_train = self.network.select("guidance")(
            self._actor_input(batch["observations"][0], x_t, t),
            params=grad_params,
        )
        distill_loss = (
            jnp.square(w_train - consensus_target).sum(axis=-1) * valids
        ).mean()

        safe_w, safety = self._behavior_safe_direction(w_train, behavior_velocity)
        g_field = self.config["guidance_coef"] * t * safe_w

        # ---- Actor RL with warmup/ramp (avoid 0*NaN via lax.cond) ----
        actor_weight = self.actor_weight_from_step(
            self.network.step,
            self.config["bc_warmup_steps"],
            self.config["actor_ramp_steps"],
            self.config["actor_coef"],
        )

        def _actor_branch(_):
            # v_theta NOT stop-grad; G is stop-grad; decode WITHOUT stop-grad;
            # Q via self.network (fixed params) so grads flow to actor/decoder only.
            guided_v = behavior_velocity + jax.lax.stop_gradient(g_field)
            dt = jnp.minimum(1.0 / n_steps, 1.0 - t)
            x_plus = self.project_box(
                x_t + guided_v * dt,
                self.latent_box_min,
                self.latent_box_max,
            )
            t_plus = jnp.clip(t + 1.0 / n_steps, 0.0, 1.0)
            y_plus = self._decode_flat_latent(x_plus)
            q_pe = self.network.select("value")(
                jnp.concatenate([batch["observations"][0], y_plus, t_plus], axis=-1)
            )
            q_pe = q_pe.mean(axis=0)
            actor_loss = -(q_pe * valids).mean()
            return actor_loss, q_pe

        def _skip_actor(_):
            zero = jnp.zeros((), dtype=bc_loss.dtype)
            q_pe_zero = jnp.zeros((batch_size,), dtype=bc_loss.dtype)
            return zero, q_pe_zero

        actor_loss, q_pe = jax.lax.cond(
            actor_weight > 0.0,
            _actor_branch,
            _skip_actor,
            operand=None,
        )

        total_loss = (
            actor_weight * actor_loss
            + self.config["alpha"] * bc_loss
            + critic_loss
            + self.config["distill_coef"] * distill_loss
        )

        actions_flat = rearrange(actions_btd, "b h d -> b (h d)")
        clean_recons = self._decode_flat_latent(x1)
        decode_rmse = jnp.sqrt(jnp.mean(jnp.square(clean_recons - actions_flat)))

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "actor_weight": actor_weight,
            "bc_loss": bc_loss,
            "bc_weight_mean": bc_weight.mean(),
            "critic_loss": critic_loss,
            "distill_loss": distill_loss,
            "decode_rmse": decode_rmse,
            "q": q.mean(),
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "q_pe_mean": q_pe.mean(),
            "q_pe_max": q_pe.max(),
            "q_pe_min": q_pe.min(),
            "w_norm": jnp.linalg.norm(w_train, axis=-1).mean(),
            "safe_w_norm": jnp.linalg.norm(safe_w, axis=-1).mean(),
            "consensus_target_norm": jnp.linalg.norm(consensus_target, axis=-1).mean(),
            "q_grad_norm": q_grad_norm.mean(),
            "q_grad_scale": grad_scale,
            "guidance_coef_eff": self.config["guidance_coef"],
            "latent_x0_norm": jnp.linalg.norm(x0, axis=-1).mean(),
            "latent_xt_norm": jnp.linalg.norm(x_t, axis=-1).mean(),
            "behavior_alignment_cos": safety["behavior_alignment_cos"].mean(),
            "behavior_conflict_fraction": safety["behavior_conflict"].mean(),
            "guidance_retained": safety["guidance_retained"].mean(),
            "conflict_kill_frac": safety["conflict_kill_frac"].mean(),
            "residual_damp": safety["residual_damp"].mean(),
            "trust": safety["trust"].mean(),
        }

    def target_update(self, network, module_name, d):
        """Polyak/EMA targets from *post-update* online params (v4-only fix)."""
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
        self.target_update(new_network, "value", d=self.config["tau"])
        self.target_update(new_network, "actor", d=1 - self.config["ema"])
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(self, observations, noise_latent, seed=None, temperature=0.0):
        """N guided Euler steps in latent box from uniform-token source; decode.

        ``seed`` is accepted for API compatibility with sibling agents but is
        unused: Euler integration here is deterministic given ``noise_latent``.
        """
        del seed
        actions_latent = noise_latent
        actor_name = "actor" if temperature > 0 else "target_actor"
        n = self.config["flow_steps"]
        for i in range(n):
            t = jnp.full((*observations.shape[:-1], 1), i / n)
            fm = self._actor_input(observations, actions_latent, t)
            behavior_velocity = self.network.select(actor_name)(fm)
            guidance, _, _, _ = self.guidance_field(
                observations,
                actions_latent,
                t,
                behavior_velocity,
                stop_guidance=False,
            )
            actions_latent = self.project_box(
                actions_latent + (behavior_velocity + guidance) / n,
                self.latent_box_min,
                self.latent_box_max,
            )
        actions = self._decode_flat_latent(actions_latent)
        return jnp.clip(actions, -1, 1)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        """Sample actions; ``seed`` only draws the uniform-token source latent."""
        obs = jnp.atleast_2d(obs)[-1:]
        noise_latent, _ = self.sample_uniform_token_latent(
            seed, 1, self.config["num_registers"]
        )
        actions = self.compute_flow_actions(
            obs,
            noise_latent=noise_latent,
            temperature=temperature,
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
        latent_dim = num_registers * FSQ_DIM

        tokenizer_path = config.get("tokenizer_path", "")
        if not tokenizer_path:
            raise ValueError(
                "ConsensusLatentFlow requires agent.tokenizer_path to a "
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
        latent_dim = num_registers * FSQ_DIM
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
        ex_latent = jnp.zeros((ex_obs.shape[0], latent_dim), dtype=jnp.float32)
        ex_actor_in = jnp.concatenate([ex_obs, ex_latent, ex_times], axis=-1)
        ex_value_in = jnp.concatenate([ex_obs, ex_flat_actions, ex_times], axis=-1)

        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )
        actor_def = LatentFlowActor(
            hidden_dims=config["actor_hidden_dims"],
            latent_dim=latent_dim,
            layer_norm=config["layer_norm"],
        )
        guidance_def = LatentGuidance(
            hidden_dims=config["guidance_hidden_dims"],
            latent_dim=latent_dim,
            layer_norm=config["layer_norm"],
        )

        network_info = dict(
            value=(value_def, (ex_value_in,)),
            target_value=(copy.deepcopy(value_def), (ex_value_in,)),
            actor=(actor_def, (ex_actor_in,)),
            target_actor=(copy.deepcopy(actor_def), (ex_actor_in,)),
            guidance=(guidance_def, (ex_actor_in,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]
        params["modules_target_actor"] = params["modules_actor"]

        config["action_dim"] = action_dim
        config["prim_action_dim"] = prim_dim
        config["latent_dim"] = latent_dim
        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(h)) + [jnp.inf])
        )
        codebook = build_codebook(FSQ_LEVELS)
        box_min, box_max = cls.codebook_box_bounds(codebook, num_registers)

        return cls(
            rng=rng,
            network=network,
            tokenizer_def=tokenizer_def,
            tokenizer_params=tok_params,
            codebook=codebook,
            latent_box_min=box_min,
            latent_box_max=box_max,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="consensus_latent_flow",
            h=3,
            alpha=1.0,
            actor_coef=1.0,
            bc_warmup_steps=50_000,
            actor_ramp_steps=50_000,
            expectile=0.5,
            ensemble_ct=10,
            rho=0.0,
            lr=3e-4,
            discount=0.99,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            guidance_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            tau=0.005,
            ema=0.999,
            flow_steps=10,
            guidance_coef=0.5,
            distill_coef=1.0,
            consensus_floor=0.01,
            conflict_power=2.0,
            residual_coef=0.25,
            num_registers=12,
            tokenizer_path="",
        )
    )
    return config
