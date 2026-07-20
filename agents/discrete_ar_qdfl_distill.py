"""Paired QuantizedDFLRQL9 teacher + CausalAR student (token distillation).

Both start randomly and co-train on the same offline batch. The student
distills online teacher EMA tokens (target_actor + guidance → frozen OATTok
encode, stop-grad) plus a configurable light dataset-token BC anchor.

``freeze_teacher`` (static config) skips the teacher update entirely for
phase-2 student-only continuation. Orchestration is Python-side calling
individually jitted teacher update / token sampling / student update to
avoid nested-JIT issues.

Checkpoint pytree preserves teacher + student TrainStates. ``network`` is
the student TrainState so ``main.py`` resume via ``agent.network.step`` works.
Deploy / eval uses the AR student only.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange

from agents.discrete_ar_iql import (
    BOS_ID,
    CausalARActor,
    DiscreteARIQLAgent,
)
from agents.oattok_jax import (
    CODEBOOK_SIZE,
    FSQ_LEVELS,
    OATTok,
    build_codebook,
    indices_to_codes,
    load_tokenizer,
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


class DiscreteARQdflDistillAgent(flax.struct.PyTreeNode):
    """QuantizedDFLRQL9 teacher + Causal AR token student."""

    rng: Any
    teacher: QuantizedDFLRQL9Agent
    network: Any  # student TrainState (actor + target_actor)
    tokenizer_def: Any = nonpytree_field()
    tokenizer_params: Any = nonpytree_field()
    codebook: Any = nonpytree_field()
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Token / log-prob utilities (exact AR factorization)
    # ------------------------------------------------------------------

    @staticmethod
    def make_teacher_inputs(tokens, bos_id: int = BOS_ID):
        return DiscreteARIQLAgent.make_teacher_inputs(tokens, bos_id=bos_id)

    @staticmethod
    def token_ce(logits, targets):
        return DiscreteARIQLAgent.token_ce(logits, targets)

    @staticmethod
    def token_log_probs_from_logits(logits, tokens):
        """Exact per-token log π(z_i | s, z_<i); logits (B,K,V), tokens (B,K)."""
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(
            log_probs, tokens[..., None].astype(jnp.int32), axis=-1
        )[..., 0]

    @staticmethod
    def sequence_log_probs_from_logits(logits, tokens):
        """Exact sequence log π(z|s) = sum_i log π(z_i|s,z_<i); shape (B,)."""
        return DiscreteARQdflDistillAgent.token_log_probs_from_logits(
            logits, tokens
        ).sum(axis=-1)

    def _encode_actions(self, actions_btd):
        """(B, T, D) → tokens (B, K), codes (B, K, q)."""
        quant, tokens = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            actions_btd,
            method=OATTok.encode,
            deterministic=True,
        )
        return tokens, quant

    def _decode_codes(self, codes):
        """(B, K, q) → flattened continuous action (B, h*Da)."""
        recons = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            codes,
            method=OATTok.decode,
            deterministic=True,
        )
        return rearrange(recons, "b t d -> b (t d)")

    def _actor_logits(self, observations, token_inputs, actor_name, params=None):
        return self.network.select(actor_name)(
            observations, token_inputs, params=params
        )

    def compute_token_log_probs(
        self,
        observations,
        tokens,
        actor_name: str = "actor",
        params=None,
    ):
        """Teacher-forced exact per-token log-probs; shape (B, K)."""
        token_inputs = self.make_teacher_inputs(tokens, BOS_ID)
        logits = self._actor_logits(
            observations, token_inputs, actor_name, params=params
        )
        return self.token_log_probs_from_logits(logits, tokens)

    def compute_sequence_log_probs(
        self,
        observations,
        tokens,
        actor_name: str = "actor",
        params=None,
    ):
        """Teacher-forced exact sequence log-probs; shape (B,)."""
        return self.compute_token_log_probs(
            observations, tokens, actor_name=actor_name, params=params
        ).sum(axis=-1)

    @partial(jax.jit, static_argnames=("actor_name",))
    def token_log_probs(self, observations, tokens, actor_name: str = "actor"):
        """Jitted exact per-token log-probs API for later PPO."""
        return self.compute_token_log_probs(
            observations, tokens, actor_name=actor_name
        )

    @partial(jax.jit, static_argnames=("actor_name",))
    def sequence_log_probs(self, observations, tokens, actor_name: str = "actor"):
        """Jitted exact sequence log-probs API for later PPO."""
        return self.compute_sequence_log_probs(
            observations, tokens, actor_name=actor_name
        )

    # ------------------------------------------------------------------
    # Student loss / update (jitted; teacher tokens are stop-grad inputs)
    # ------------------------------------------------------------------

    @jax.jit
    def student_total_loss(self, batch, teacher_tokens, grad_params, rng=None):
        """Distill CE to teacher tokens + light dataset-token BC.

        No student critics / SS / Q-actor. ``teacher_tokens`` must already be
        stop-grad'd by the teacher sampler.
        """
        del rng  # Reserved for future stochastic student regularizers.
        h = int(self.config["h"])
        k = int(self.config["num_registers"])
        distill_coef = jnp.asarray(
            self.config["distill_coef"], dtype=jnp.float32
        )
        bc_coef = jnp.asarray(self.config["bc_coef"], dtype=jnp.float32)

        observations = batch["observations"][0]
        actions_btd = rearrange(batch["actions"][:h], "h b d -> b h d")
        dataset_tokens, _ = self._encode_actions(actions_btd)
        dataset_tokens = jax.lax.stop_gradient(dataset_tokens.astype(jnp.int32))
        teacher_tokens = jax.lax.stop_gradient(teacher_tokens.astype(jnp.int32))

        # Optional chunk validity (post-terminal CE weight → 0).
        terminals = batch["terminals"]
        rs_terminals = jnp.concatenate(
            [jnp.zeros_like(terminals[:1]), terminals[:-1]],
            axis=0,
        )
        chunk_w = DiscreteARIQLAgent.chunk_ce_weights(rs_terminals, h)
        weight_sum = jnp.maximum(chunk_w.sum(), 1e-6)

        # Distill: teacher-forced CE to online teacher tokens.
        distill_inputs = self.make_teacher_inputs(teacher_tokens, BOS_ID)
        distill_logits = self._actor_logits(
            observations, distill_inputs, "actor", params=grad_params
        )
        distill_ce_tok = self.token_ce(distill_logits, teacher_tokens)
        distill_ce_ex = distill_ce_tok.mean(axis=-1)
        distill_loss = (distill_ce_ex * chunk_w).sum() / weight_sum

        # Light dataset-token BC anchor.
        bc_inputs = self.make_teacher_inputs(dataset_tokens, BOS_ID)
        bc_logits = self._actor_logits(
            observations, bc_inputs, "actor", params=grad_params
        )
        bc_ce_tok = self.token_ce(bc_logits, dataset_tokens)
        bc_ce_ex = bc_ce_tok.mean(axis=-1)
        bc_loss = (bc_ce_ex * chunk_w).sum() / weight_sum

        total_loss = distill_coef * distill_loss + bc_coef * bc_loss

        distill_pred = jnp.argmax(distill_logits, axis=-1)
        bc_pred = jnp.argmax(bc_logits, axis=-1)
        distill_token_acc = (distill_pred == teacher_tokens).astype(jnp.float32).mean()
        bc_token_acc = (bc_pred == dataset_tokens).astype(jnp.float32).mean()
        distill_seq_exact = (
            (distill_pred == teacher_tokens).all(axis=-1).astype(jnp.float32).mean()
        )
        teacher_vs_data_agree = (
            (teacher_tokens == dataset_tokens).astype(jnp.float32).mean()
        )

        # Exact log-prob diagnostics under current params (stop-grad labels).
        distill_tok_lp = self.token_log_probs_from_logits(
            distill_logits, teacher_tokens
        )
        distill_seq_lp = distill_tok_lp.sum(axis=-1)

        info = {
            "total_loss": total_loss,
            "distill_loss": distill_loss,
            "bc_loss": bc_loss,
            "distill_coef": distill_coef,
            "bc_coef": bc_coef,
            "distill_token_acc": distill_token_acc,
            "bc_token_acc": bc_token_acc,
            "distill_seq_exact": distill_seq_exact,
            "teacher_vs_data_token_agree": teacher_vs_data_agree,
            "distill_token_logprob_mean": distill_tok_lp.mean(),
            "distill_seq_logprob_mean": distill_seq_lp.mean(),
            "num_registers": jnp.asarray(float(k)),
            "vocab_size": jnp.asarray(float(CODEBOOK_SIZE)),
            "freeze_teacher": jnp.asarray(
                float(bool(self.config["freeze_teacher"]))
            ),
        }
        return total_loss, info

    def target_update(self, network, module_name, d):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def student_update(self, batch, teacher_tokens):
        """One jitted student step; does not touch the teacher."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.student_total_loss(
                batch, teacher_tokens, grad_params, rng=rng
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # ``ema`` is a nonpytree config float; bake into Python before JIT body.
        ema = self.config["ema"]
        self.target_update(new_network, "actor", d=1.0 - ema)
        return self.replace(network=new_network, rng=new_rng), info

    def update(self, batch):
        """Python orchestration: teacher update → token sample → student update.

        ``freeze_teacher`` is read as a static Python bool and skips the teacher
        ``update`` entirely (no teacher gradients / step bump).
        """
        freeze_teacher = bool(self.config["freeze_teacher"])
        teacher = self.teacher
        teacher_info: dict = {}

        if not freeze_teacher:
            teacher, teacher_info = teacher.update(batch)

        rng, sample_rng = jax.random.split(self.rng)
        observations = batch["observations"][0]
        teacher_temp = float(self.config.get("teacher_temperature", 0.0))
        teacher_tokens = teacher.sample_tokens_batch(
            observations,
            seed=sample_rng,
            temperature=teacher_temp,
        )

        # Replace rng before student_update so the student consumes a fresh key.
        agent = self.replace(rng=rng, teacher=teacher)
        agent, student_info = agent.student_update(batch, teacher_tokens)

        info = merge_infos(
            prefix_info(teacher_info, "teacher_"),
            student_info,
            {
                "freeze_teacher": jnp.asarray(float(freeze_teacher)),
            },
        )
        return agent, info

    # ------------------------------------------------------------------
    # Deploy: AR freerun → frozen decode (student only)
    # ------------------------------------------------------------------

    def _resolve_actor_and_sampling_temp(
        self, temperature: float
    ) -> Tuple[str, float]:
        if temperature > 0.0:
            return "actor", float(temperature)
        return "target_actor", float(self.config["eval_sampling_temperature"])

    def _sample_token(self, logits, rng, temperature):
        if temperature == 0.0:
            return jnp.argmax(logits, axis=-1).astype(jnp.int32), rng
        rng, s_rng = jax.random.split(rng)
        sample_logits = logits / jnp.maximum(temperature, 1e-6)
        ids = jax.random.categorical(s_rng, sample_logits).astype(jnp.int32)
        return ids, rng

    def _ar_fill_tokens(self, observations, actor_name, sampling_temp, rng):
        b = observations.shape[0]
        k = int(self.config["num_registers"])
        tokens = jnp.zeros((b, k), dtype=jnp.int32)
        for i in range(k):
            rng, step_rng = jax.random.split(rng)
            inp = jnp.full((b, k), BOS_ID, dtype=jnp.int32)
            if i > 0:
                inp = inp.at[:, 1 : i + 1].set(tokens[:, :i])
            logits = self._actor_logits(observations, inp, actor_name)
            next_ids, _ = self._sample_token(
                logits[:, i, :], step_rng, sampling_temp
            )
            tokens = tokens.at[:, i].set(next_ids)
        return tokens, rng

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_ar_actions(self, observations, seed=None, temperature=0.0):
        """Autoregressive K-step decode via frozen OATTok."""
        actor_name, sampling_temp = self._resolve_actor_and_sampling_temp(
            temperature
        )
        rng = seed if seed is not None else self.rng
        tokens, _ = self._ar_fill_tokens(
            observations, actor_name, sampling_temp, rng
        )
        codes = indices_to_codes(tokens, FSQ_LEVELS)
        actions = self._decode_codes(codes)
        return jnp.clip(actions, -1.0, 1.0)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        """Student AR deploy; shape ``(h, Da)``."""
        obs = jnp.atleast_2d(obs)[-1:]
        actions = self.compute_ar_actions(
            obs, seed=seed, temperature=temperature
        )[0]
        return rearrange(actions, "(h d) -> h d", h=self.config["h"])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def validate_create_config(config):
        emb_dim = int(config["actor_emb_dim"])
        num_heads = int(config["actor_num_heads"])
        if num_heads <= 0:
            raise ValueError(f"actor_num_heads must be positive, got {num_heads}")
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"actor_emb_dim ({emb_dim}) must be divisible by "
                f"actor_num_heads ({num_heads})"
            )
        dropout = float(config.get("actor_dropout", 0.0))
        if dropout != 0.0:
            raise ValueError(
                f"actor_dropout={dropout} is not supported (must be 0.0)."
            )
        if float(config.get("distill_coef", 0.0)) < 0.0:
            raise ValueError("distill_coef must be >= 0")
        if float(config.get("bc_coef", 0.0)) < 0.0:
            raise ValueError("bc_coef must be >= 0")
        if not config.get("tokenizer_path"):
            raise ValueError(
                "discrete_ar_qdfl_distill requires agent.tokenizer_path"
            )

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create random teacher + random AR student sharing frozen OATTok."""
        config = dict(config)
        cls.validate_create_config(config)

        teacher_cfg = build_teacher_config(config, dict(get_quantized_config()))
        # Distinct PRNG streams: teacher uses ``seed``, student uses ``seed+1``.
        teacher = QuantizedDFLRQL9Agent.create(
            int(seed), ex_observations, ex_actions, teacher_cfg
        )

        rng = jax.random.PRNGKey(int(seed) + 1_000_003)
        rng, init_rng = jax.random.split(rng)

        ex_actions_arr = jnp.asarray(ex_actions)
        ex_observations_arr = jnp.asarray(ex_observations)
        if ex_actions_arr.ndim == 3:
            prim_dim = int(ex_actions_arr.shape[-1])
            ex_obs = ex_observations_arr[0]
            ex_act_step = ex_actions_arr[0]
        elif ex_actions_arr.ndim == 2:
            prim_dim = int(ex_actions_arr.shape[-1])
            ex_obs = (
                ex_observations_arr
                if ex_observations_arr.ndim == 2
                else ex_observations_arr[None, :]
            )
            ex_act_step = ex_actions_arr
        else:
            prim_dim = int(ex_actions_arr.shape[0])
            ex_obs = (
                ex_observations_arr[None, :]
                if ex_observations_arr.ndim == 1
                else ex_observations_arr
            )
            ex_act_step = ex_actions_arr[None, :]

        del ex_act_step  # Used only for dim inference above.
        h = int(config["h"])
        action_dim = prim_dim * h

        # Prefer tokenizer already attached to the teacher (same path/meta).
        tokenizer_def = teacher.tokenizer_def
        tokenizer_params = teacher.tokenizer_params
        tok_meta = teacher.tokenizer_meta
        if tokenizer_def is None:
            tok_params, tok_meta = load_tokenizer(config["tokenizer_path"])
            QuantizedDFLRQL9Agent.assert_tokenizer_matches_policy(
                tok_meta, h, prim_dim
            )
            tokenizer_def = QuantizedDFLRQL9Agent.build_oattok(tok_meta)
            tokenizer_params = tok_params

        num_registers = int(
            tok_meta.get("num_registers", config.get("num_registers", 16))
        )
        config["num_registers"] = num_registers
        config["action_dim"] = action_dim
        config["prim_action_dim"] = prim_dim
        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(h)) + [jnp.inf])
        )

        ex_token_inputs = jnp.full(
            (ex_obs.shape[0], num_registers), BOS_ID, dtype=jnp.int32
        )
        actor_def = CausalARActor(
            emb_dim=int(config["actor_emb_dim"]),
            depth=int(config["actor_depth"]),
            num_heads=int(config["actor_num_heads"]),
            vocab_size=CODEBOOK_SIZE,
            num_registers=num_registers,
            dropout=0.0,
        )
        network_info = dict(
            actor=(actor_def, (ex_obs, ex_token_inputs)),
            target_actor=(copy.deepcopy(actor_def), (ex_obs, ex_token_inputs)),
        )
        networks = {name: spec[0] for name, spec in network_info.items()}
        network_args = {name: spec[1] for name, spec in network_info.items()}
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
            tokenizer_def=tokenizer_def,
            tokenizer_params=tokenizer_params,
            codebook=codebook,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    """Paired AR distill defaults (HL humanoidmaze-friendly)."""
    teacher_defaults = get_quantized_config()
    config = mlc.ConfigDict(
        dict(
            agent_name="discrete_ar_qdfl_distill",
            # Shared / teacher geometry.
            h=1,
            batch_size=256,
            lr=3e-4,
            discount=0.995,
            alpha=0.3,
            expectile=0.5,
            ensemble_ct=10,
            rho=0.0,
            flow_steps=10,
            guidance_coef=0.5,
            teacher_distill_coef=float(
                teacher_defaults.get("distill_coef", 1.0)
            ),
            consensus_floor=0.01,
            conflict_power=2.0,
            residual_coef=0.25,
            actor_hidden_dims=tuple(
                teacher_defaults.get(
                    "actor_hidden_dims", (512, 512, 512, 512)
                )
            ),
            value_hidden_dims=tuple(
                teacher_defaults.get(
                    "value_hidden_dims", (512, 512, 512, 512)
                )
            ),
            guidance_hidden_dims=tuple(
                teacher_defaults.get(
                    "guidance_hidden_dims", (512, 512, 512, 512)
                )
            ),
            layer_norm=True,
            actor_layer_norm=False,
            tau=0.005,
            ema=0.999,
            tokenizer_path="",
            projection_enabled=True,
            # Phase control: True → skip teacher.update entirely.
            freeze_teacher=False,
            # Student distillation.
            distill_coef=1.0,
            bc_coef=0.1,
            teacher_temperature=0.0,  # 0 → EMA target_actor + guidance
            # Causal AR student.
            actor_emb_dim=256,
            actor_depth=4,
            actor_num_heads=4,
            actor_dropout=0.0,
            num_registers=16,
            eval_sampling_temperature=1.0,
        )
    )
    return config
