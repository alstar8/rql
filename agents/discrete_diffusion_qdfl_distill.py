"""Paired QuantizedDFLRQL9 teacher + categorical diffusion student.

Both start randomly (frozen OATTok only). On each offline batch the teacher
runs its unchanged QuantizedDFLRQL9 update (unless ``freeze_teacher``), then the
student distills stop-grad OATTok tokens from the teacher's EMA guided flow
via CRAFT/CDF mixture corruption and clean-token posterior CE.

No consensus guidance and no student critic. Deploy is iterative categorical
denoising + one frozen decode. ``sample_tokens_with_logprob`` and trajectory
rescoring expose exact mixture-path transition log-probs for later DDPO.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any, Dict

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange

from agents.consensus_discrete_flow import (
    ConsensusDiscreteFlowAgent,
    DiscreteFlowActor,
)
from agents.oattok_jax import (
    CODEBOOK_SIZE,
    FSQ_DIM,
    FSQ_LEVELS,
    OATTok,
    build_codebook,
    indices_to_codes,
)
from agents.paired_qdfl_helpers import (
    build_teacher_config,
    merge_infos,
    prefix_info,
)
from agents.quantized_dflrql9 import (
    QuantizedDFLRQL9Agent,
    get_config as get_quantized_config,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


# ---------------------------------------------------------------------------
# Uniquely named mixture-path log-prob helpers (DDPO / rescoring).
# ---------------------------------------------------------------------------


def qdfl_dd_mixture_replace_probability(t, dt):
    """Bernoulli replace probability on the CRAFT/CDF mixture path."""
    return ConsensusDiscreteFlowAgent.mixture_replace_probability(t, dt)


def qdfl_dd_path_step_logprob(
    tokens,
    next_tokens,
    logits,
    replace_mask,
    t,
    dt,
    force_replace=False,
):
    """Exact path log-prob for one mixture step given stored replace decisions.

    For each register:
      replaced -> log p_rep + log Cat(next | logits)
      kept     -> log(1 - p_rep)
    With ``force_replace=True``, p_rep is treated as 1 (final commit).

    Shapes: tokens/next/replace ``(..., K)``; logits ``(..., K, V)``.
    Returns ``(...,)`` summed over registers.
    """
    log_cat = jax.nn.log_softmax(logits, axis=-1)
    next_oh = jax.nn.one_hot(
        next_tokens, logits.shape[-1], dtype=log_cat.dtype
    )
    log_p_tok = (log_cat * next_oh).sum(axis=-1)

    if force_replace is True:
        return log_p_tok.sum(axis=-1)

    p_rep = qdfl_dd_mixture_replace_probability(t, dt)
    # Broadcast p over register axis when t is (..., 1).
    while p_rep.ndim < tokens.ndim:
        p_rep = p_rep[..., None]
    p_rep = jnp.broadcast_to(p_rep, tokens.shape)
    p_rep = jnp.where(
        jnp.asarray(force_replace, dtype=bool),
        jnp.ones_like(p_rep),
        p_rep,
    )
    log_rep = jnp.log(jnp.clip(p_rep, 1e-8, 1.0))
    log_keep = jnp.log(jnp.clip(1.0 - p_rep, 1e-8, 1.0))
    per = jnp.where(replace_mask, log_rep + log_p_tok, log_keep)
    return per.sum(axis=-1)


def qdfl_dd_marginal_step_logprob(
    tokens,
    next_tokens,
    logits,
    t,
    dt,
    force_replace=False,
):
    """Exact marginal P(next|tokens) under mixture keep/replace (no mask).

    P(x'|x) = (1-p)*1[x'=x] + p*Cat(x'|L), with p=1 under force_replace.
    """
    log_cat = jax.nn.log_softmax(logits, axis=-1)
    next_oh = jax.nn.one_hot(
        next_tokens, logits.shape[-1], dtype=log_cat.dtype
    )
    log_p_tok = (log_cat * next_oh).sum(axis=-1)
    same = next_tokens == tokens

    if force_replace is True:
        return log_p_tok.sum(axis=-1)

    p_rep = qdfl_dd_mixture_replace_probability(t, dt)
    while p_rep.ndim < tokens.ndim:
        p_rep = p_rep[..., None]
    p_rep = jnp.broadcast_to(p_rep, tokens.shape)
    p_rep = jnp.where(
        jnp.asarray(force_replace, dtype=bool),
        jnp.ones_like(p_rep),
        p_rep,
    )
    # log((1-p) + p * Cat) when kept equal; log(p * Cat) when changed.
    cat_prob = jnp.exp(log_p_tok)
    kept_prob = (1.0 - p_rep) + p_rep * cat_prob
    changed_prob = p_rep * cat_prob
    step_prob = jnp.where(same, kept_prob, changed_prob)
    return jnp.log(jnp.clip(step_prob, 1e-12, 1.0)).sum(axis=-1)


class DiscreteDiffusionQdflDistillAgent(flax.struct.PyTreeNode):
    """Paired QuantizedDFLRQL9 teacher + categorical diffusion student."""

    rng: Any
    teacher: Any
    network: Any
    codebook: Any = nonpytree_field()
    config: Any = nonpytree_field()

    # ---- tokenizer / actor helpers (reuse teacher's frozen OATTok) ----

    def _encode_actions_btd(self, actions_btd):
        quant, tokens = self.teacher.tokenizer_def.apply(
            {"params": self.teacher.tokenizer_params},
            actions_btd,
            method=OATTok.encode,
            deterministic=True,
        )
        return tokens, quant

    def _encode_flat_actions(self, flat_actions):
        h = int(self.config["h"])
        actions_btd = rearrange(flat_actions, "b (h d) -> b h d", h=h)
        tokens, quant = self._encode_actions_btd(actions_btd)
        return tokens, quant

    def _decode_codes(self, codes):
        recons = self.teacher.tokenizer_def.apply(
            {"params": self.teacher.tokenizer_params},
            codes,
            method=OATTok.decode,
            deterministic=True,
        )
        return rearrange(recons, "b t d -> b (t d)")

    def _decode_tokens(self, tokens):
        codes = indices_to_codes(tokens, FSQ_LEVELS)
        return self._decode_codes(codes)

    def _actor_input(self, observations, codes, times):
        flat_codes = rearrange(codes, "b k q -> b (k q)")
        return jnp.concatenate([observations, flat_codes, times], axis=-1)

    def _soft_targets(self, tokens):
        codebook = self.codebook
        true_codes = indices_to_codes(tokens, FSQ_LEVELS)
        diff = true_codes[:, :, None, :] - codebook[None, None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)
        neighbor = jax.nn.softmax(
            -dist2 / self.config["soft_target_temperature"], axis=-1
        )
        one_hot = jax.nn.one_hot(tokens, CODEBOOK_SIZE)
        eps = self.config["soft_target_eps"]
        return (1.0 - eps) * one_hot + eps * neighbor

    @staticmethod
    def _corrupt_tokens(rng, clean_tokens, t):
        return ConsensusDiscreteFlowAgent._corrupt_tokens(rng, clean_tokens, t)

    # ---- teacher token targets (EMA flow → OATTok IDs) ----

    def sample_teacher_tokens(self, observations, seed):
        """Batched EMA guided flow → frozen OATTok token IDs (stop-grad).

        Uses teacher ``sample_tokens_batch`` (target_actor + guidance at
        ``teacher_temperature``, default 0) from Python orchestration to
        avoid nested JIT.
        """
        observations = jnp.asarray(observations)
        if observations.ndim == 1:
            observations = observations[None, :]
        teacher_temp = float(self.config.get("teacher_temperature", 0.0))
        sampler = getattr(self.teacher, "sample_tokens_batch", None)
        if callable(sampler):
            return sampler(
                observations, seed=seed, temperature=teacher_temp
            )
        # Fallback if teacher lacks the batched sampler (self-contained).
        action_rng, n_rng = jax.random.split(seed)
        noise = jax.random.normal(
            n_rng, (observations.shape[0], int(self.teacher.config["action_dim"]))
        )
        flat = self.teacher.compute_flow_actions(
            observations,
            noise=noise,
            seed=action_rng,
            temperature=teacher_temp,
        )
        tokens, _ = self._encode_flat_actions(flat)
        return jax.lax.stop_gradient(tokens)

    # ---- student CE loss ----

    def student_total_loss(self, batch, teacher_tokens, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        batch_size = int(self.config["batch_size"])
        h = int(self.config["h"])
        observations = batch["observations"][0]

        teacher_tokens = jax.lax.stop_gradient(teacher_tokens)

        rs_terminals = jnp.concatenate(
            [jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]],
            axis=0,
        )
        ce_weight = ConsensusDiscreteFlowAgent.chunk_token_ce_weights(
            rs_terminals, h
        )

        rng, t_rng, corr_rng, ds_rng, ds_t_rng, ds_corr_rng = jax.random.split(
            rng, 6
        )

        # Distill CE on corrupted teacher tokens.
        t = jax.random.uniform(t_rng, (batch_size, 1))
        t = jnp.clip(t, 1e-3, 1.0 - 1e-3)
        x_t = self._corrupt_tokens(corr_rng, teacher_tokens, t)
        x_codes = indices_to_codes(x_t, FSQ_LEVELS)
        logits = self.network.select("actor")(
            self._actor_input(observations, x_codes, t),
            params=grad_params,
        )
        soft_tgt = self._soft_targets(teacher_tokens)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce_tok = -(soft_tgt * log_probs).sum(axis=-1)
        distill_ce = (ce_tok.mean(axis=-1) * ce_weight).sum() / jnp.maximum(
            ce_weight.sum(), 1e-6
        )

        # Light dataset-token BC anchor (same mixture path).
        actions_btd = rearrange(batch["actions"][:h], "h b d -> b h d")
        dataset_tokens, _ = self._encode_actions_btd(actions_btd)
        dataset_tokens = jax.lax.stop_gradient(dataset_tokens)
        t_ds = jax.random.uniform(ds_t_rng, (batch_size, 1))
        t_ds = jnp.clip(t_ds, 1e-3, 1.0 - 1e-3)
        x_ds = self._corrupt_tokens(ds_corr_rng, dataset_tokens, t_ds)
        x_ds_codes = indices_to_codes(x_ds, FSQ_LEVELS)
        logits_ds = self.network.select("actor")(
            self._actor_input(observations, x_ds_codes, t_ds),
            params=grad_params,
        )
        soft_ds = self._soft_targets(dataset_tokens)
        log_ps_ds = jax.nn.log_softmax(logits_ds, axis=-1)
        ce_ds_tok = -(soft_ds * log_ps_ds).sum(axis=-1)
        dataset_ce = (ce_ds_tok.mean(axis=-1) * ce_weight).sum() / jnp.maximum(
            ce_weight.sum(), 1e-6
        )

        distill_coef = self.config["distill_coef"]
        dataset_bc_coef = self.config["dataset_bc_coef"]
        total_loss = distill_coef * distill_ce + dataset_bc_coef * dataset_ce

        pred = jnp.argmax(logits, axis=-1)
        correct = pred == teacher_tokens
        corrupted = x_t != teacher_tokens
        retained = jnp.logical_not(corrupted)
        decode_rmse = jnp.sqrt(
            jnp.mean(
                jnp.square(
                    self._decode_tokens(teacher_tokens)
                    - rearrange(actions_btd, "b h d -> b (h d)")
                )
            )
        )

        return total_loss, {
            "total_loss": total_loss,
            "distill_ce": distill_ce,
            "dataset_ce": dataset_ce,
            "token_acc": correct.mean(),
            "token_acc_corrupted": ConsensusDiscreteFlowAgent.safe_masked_mean(
                correct, corrupted
            ),
            "token_acc_retained": ConsensusDiscreteFlowAgent.safe_masked_mean(
                correct, retained
            ),
            "corruption_frac": corrupted.mean(),
            "ce_weight_mean": ce_weight.mean(),
            "teacher_decode_vs_data_rmse": decode_rmse,
            "teacher_token_mean": teacher_tokens.astype(jnp.float32).mean(),
        }

    def target_update(self, network, module_name, d):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def _student_update(self, batch, teacher_tokens):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.student_total_loss(
                batch, teacher_tokens, grad_params, rng=rng
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "actor", d=1.0 - self.config["ema"])
        return self.replace(network=new_network, rng=new_rng), info

    def update(self, batch):
        """Python-orchestrated teacher update → teacher tokens → student update.

        ``freeze_teacher`` (static config) skips the teacher update entirely so
        phase-2 student-only training does not touch teacher grads/params.
        """
        freeze_teacher = bool(self.config.get("freeze_teacher", False))
        teacher = self.teacher
        teacher_info: Dict[str, Any] = {}

        if not freeze_teacher:
            teacher, teacher_info = teacher.update(batch)

        rng, sample_rng = jax.random.split(self.rng)
        teacher_tokens = self.replace(teacher=teacher).sample_teacher_tokens(
            batch["observations"][0], sample_rng
        )
        agent = self.replace(rng=rng, teacher=teacher)
        agent, student_info = agent._student_update(batch, teacher_tokens)
        info = merge_infos(
            prefix_info(teacher_info, "teacher_"),
            student_info,
            {"freeze_teacher": jnp.asarray(float(freeze_teacher))},
        )
        return agent, info

    # ---- sampling / DDPO log-probs ----

    def _posterior_logits(self, observations, tokens, t, temperature, actor_name):
        codes = indices_to_codes(tokens, FSQ_LEVELS)
        logits = self.network.select(actor_name)(
            self._actor_input(observations, codes, t)
        )
        if temperature is not None and temperature > 0:
            logits = logits / temperature
        return logits

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_tokens_with_logprob(self, observations, seed, temperature=1.0):
        """Iterative mixture denoising with exact path transition log-probs.

        Returns a dict:
          tokens: (B, K) final tokens
          token_trajectory: (N+1, B, K)
          replace_masks: (N, B, K) bool
          step_logprobs: (N, B) path log-probs per step
          logprob: (B,) sum over steps
        """
        observations = jnp.atleast_2d(observations)
        batch_size = observations.shape[0]
        k = int(self.config["num_registers"])
        n = int(self.config["flow_steps"])
        dt = 1.0 / n
        actor_name = "actor" if temperature > 0 else "target_actor"
        sample_temp = (
            float(temperature)
            if temperature and temperature > 0
            else float(self.config.get("eval_sampling_temperature", 1.0))
        )

        rng, n_rng = jax.random.split(seed)
        tokens = jax.random.randint(n_rng, (batch_size, k), 0, CODEBOOK_SIZE)

        traj = [tokens]
        replace_masks = []
        step_logprobs = []

        for i in range(n):
            rng, step_rng = jax.random.split(rng)
            t = jnp.full((batch_size, 1), i / n)
            force = i == n - 1
            logits = self._posterior_logits(
                observations, tokens, t, sample_temp, actor_name
            )
            cat_rng, bern_rng = jax.random.split(step_rng)
            candidates = jax.random.categorical(cat_rng, logits).astype(
                tokens.dtype
            )
            if force:
                replace = jnp.ones(tokens.shape, dtype=bool)
                next_tokens = candidates
            else:
                p_rep = qdfl_dd_mixture_replace_probability(t, dt)
                replace = jax.random.uniform(bern_rng, tokens.shape) < p_rep
                next_tokens = jnp.where(replace, candidates, tokens)
            lp = qdfl_dd_path_step_logprob(
                tokens,
                next_tokens,
                logits,
                replace,
                t,
                dt,
                force_replace=force,
            )
            tokens = next_tokens
            traj.append(tokens)
            replace_masks.append(replace)
            step_logprobs.append(lp)

        token_trajectory = jnp.stack(traj, axis=0)
        replace_masks_arr = jnp.stack(replace_masks, axis=0)
        step_logprobs_arr = jnp.stack(step_logprobs, axis=0)
        return {
            "tokens": tokens,
            "token_trajectory": token_trajectory,
            "replace_masks": replace_masks_arr,
            "step_logprobs": step_logprobs_arr,
            "logprob": step_logprobs_arr.sum(axis=0),
        }

    @partial(jax.jit, static_argnames=("temperature", "use_path_masks"))
    def rescore_trajectory_logprob(
        self,
        observations,
        token_trajectory,
        replace_masks=None,
        temperature=1.0,
        use_path_masks=True,
    ):
        """Rescore a stored denoising trajectory under current student params.

        If ``use_path_masks`` and ``replace_masks`` are provided, uses exact path
        log-probs (DDPO). Otherwise uses marginal keep/replace transitions.
        """
        observations = jnp.atleast_2d(observations)
        n = int(self.config["flow_steps"])
        dt = 1.0 / n
        actor_name = "actor" if temperature > 0 else "target_actor"
        sample_temp = (
            float(temperature)
            if temperature and temperature > 0
            else float(self.config.get("eval_sampling_temperature", 1.0))
        )

        step_lps = []
        for i in range(n):
            t = jnp.full((observations.shape[0], 1), i / n)
            force = i == n - 1
            tokens = token_trajectory[i]
            next_tokens = token_trajectory[i + 1]
            logits = self._posterior_logits(
                observations, tokens, t, sample_temp, actor_name
            )
            if use_path_masks and replace_masks is not None:
                lp = qdfl_dd_path_step_logprob(
                    tokens,
                    next_tokens,
                    logits,
                    replace_masks[i],
                    t,
                    dt,
                    force_replace=force,
                )
            else:
                lp = qdfl_dd_marginal_step_logprob(
                    tokens,
                    next_tokens,
                    logits,
                    t,
                    dt,
                    force_replace=force,
                )
            step_lps.append(lp)
        step_logprobs = jnp.stack(step_lps, axis=0)
        return {
            "step_logprobs": step_logprobs,
            "logprob": step_logprobs.sum(axis=0),
        }

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(
        self, observations, noise_tokens, seed=None, temperature=0.0
    ):
        """Mixture-path denoising (no guidance) then frozen decode."""
        tokens = noise_tokens
        actor_name = "actor" if temperature > 0 else "target_actor"
        n = int(self.config["flow_steps"])
        dt = 1.0 / n
        rng = seed if seed is not None else self.rng
        sample_temp = (
            float(temperature)
            if temperature and temperature > 0
            else float(self.config.get("eval_sampling_temperature", 1.0))
        )
        for i in range(n):
            rng, step_rng = jax.random.split(rng)
            t = jnp.full((*observations.shape[:-1], 1), i / n)
            logits = self._posterior_logits(
                observations, tokens, t, sample_temp, actor_name
            )
            tokens, _ = ConsensusDiscreteFlowAgent.posterior_mixture_update(
                tokens,
                logits,
                step_rng,
                t,
                dt,
                force_replace=(i == n - 1),
            )
        actions = self._decode_tokens(tokens)
        return jnp.clip(actions, -1.0, 1.0)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        action_rng, n_rng = jax.random.split(seed)
        obs = jnp.atleast_2d(obs)[-1:]
        noise_tokens = jax.random.randint(
            n_rng,
            (1, int(self.config["num_registers"])),
            0,
            CODEBOOK_SIZE,
        )
        actions = self.compute_flow_actions(
            obs,
            noise_tokens=noise_tokens,
            seed=action_rng,
            temperature=temperature,
        )[0]
        return rearrange(actions, "(h d) -> h d", h=int(self.config["h"]))

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        tokenizer_path = config.get("tokenizer_path", "")
        if not tokenizer_path:
            raise ValueError(
                "discrete_diffusion_qdfl_distill requires agent.tokenizer_path "
                "to a frozen OATTok .pkl."
            )

        # Independent random inits: teacher uses ``seed``, student uses fold-in.
        teacher_cfg = build_teacher_config(config, dict(get_quantized_config()))
        teacher = QuantizedDFLRQL9Agent.create(
            seed, ex_observations, ex_actions, teacher_cfg
        )

        ex_actions_arr = jnp.asarray(ex_actions)
        ex_observations_arr = jnp.asarray(ex_observations)
        if ex_actions_arr.ndim == 3:
            prim_dim = int(ex_actions_arr.shape[-1])
            ex_obs = ex_observations_arr[0]
        elif ex_actions_arr.ndim == 2:
            prim_dim = int(ex_actions_arr.shape[-1])
            ex_obs = (
                ex_observations_arr
                if ex_observations_arr.ndim == 2
                else ex_observations_arr[None, :]
            )
        else:
            prim_dim = int(ex_actions_arr.shape[0])
            ex_obs = (
                ex_observations_arr[None, :]
                if ex_observations_arr.ndim == 1
                else ex_observations_arr
            )

        h = int(config["h"])
        num_registers = int(
            teacher.tokenizer_meta.get(
                "num_registers", config.get("num_registers", 12)
            )
        )
        config["num_registers"] = num_registers
        config["prim_action_dim"] = prim_dim
        config["action_dim"] = prim_dim * h
        config.setdefault("freeze_teacher", False)
        config.setdefault("distill_coef", 1.0)
        config.setdefault("dataset_bc_coef", 0.1)
        config.setdefault("soft_target_eps", 0.1)
        config.setdefault("soft_target_temperature", 0.25)
        config.setdefault("eval_sampling_temperature", 1.0)
        config.setdefault("teacher_temperature", 0.0)
        config.setdefault("ema", 0.999)

        student_hidden = tuple(
            config.get(
                "student_actor_hidden_dims",
                config.get("actor_hidden_dims", (512, 512, 512, 512)),
            )
        )
        layer_norm = bool(config.get("layer_norm", True))

        rng = jax.random.fold_in(jax.random.PRNGKey(seed), 7919)
        rng, init_rng = jax.random.split(rng)

        ex_times = jnp.zeros((ex_obs.shape[0], 1), dtype=jnp.float32)
        ex_codes = jnp.zeros(
            (ex_obs.shape[0], num_registers, FSQ_DIM), dtype=jnp.float32
        )
        ex_actor_in = jnp.concatenate(
            [ex_obs, rearrange(ex_codes, "b k q -> b (k q)"), ex_times],
            axis=-1,
        )

        actor_def = DiscreteFlowActor(
            hidden_dims=student_hidden,
            num_registers=num_registers,
            vocab_size=CODEBOOK_SIZE,
            layer_norm=layer_norm,
        )
        network_info = dict(
            actor=(actor_def, (ex_actor_in,)),
            target_actor=(copy.deepcopy(actor_def), (ex_actor_in,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params["modules_target_actor"] = network.params["modules_actor"]

        codebook = build_codebook(FSQ_LEVELS)
        return cls(
            rng=rng,
            teacher=teacher,
            network=network,
            codebook=codebook,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = get_quantized_config()
    config.agent_name = "discrete_diffusion_qdfl_distill"
    config.freeze_teacher = False
    config.distill_coef = 1.0
    config.dataset_bc_coef = 0.1
    config.soft_target_eps = 0.1
    config.soft_target_temperature = 0.25
    config.eval_sampling_temperature = 1.0
    config.teacher_temperature = 0.0
    config.student_actor_hidden_dims = (512, 512, 512, 512)
    config.num_registers = 16
    # Student denoising steps share ``flow_steps`` with the teacher config.
    return config
