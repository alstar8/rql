"""Full MolmoAct2 + CF train loop: offline warmup (optional) + online Pick episodes.

Logs metrics every ``--log_every_episodes`` environment episodes (default 100).
Frozen MolmoAct2 serves base actions ``a_v`` with G disabled; trainable CF
modules live in the trainer process and refine actions client-side.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from buffer import StratifiedReplay, load_buffer  # noqa: E402
from models import FEATURE_DIM, MolmoAct2CF  # noqa: E402
from train_offline import (  # noqa: E402
    build_optimizers,
    critic_is_healthy,
    critic_step,
    refiner_step,
)

log = logging.getLogger("molmoact2_cf.train_full")


class OnlineReplay:
    """Growing online buffer: raw VLA features (fp16) + proprio + actions."""

    def __init__(
        self,
        model: MolmoAct2CF,
        max_transitions: int = 200_000,
        pos_frac: float = 0.4,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.max_transitions = max_transitions
        self.pos_frac = pos_frac
        self.rng = np.random.default_rng(seed)
        self.features: list[np.ndarray] = []  # float16 (FEATURE_DIM,)
        self.proprio: list[np.ndarray] = []  # float32 raw (8,)
        self.actions: list[np.ndarray] = []  # deployed, raw
        self.actions_v: list[np.ndarray] = []  # frozen VLA base action, raw
        self.returns: list[float] = []
        self.successes: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.n_episodes = 0

    def __len__(self) -> int:
        return len(self.proprio)

    def add_episode(
        self,
        proprio_raw: list[np.ndarray],
        actions_v_raw: list[np.ndarray],
        actions_raw: list[np.ndarray],
        success: bool,
        gamma: float,
        features_raw: list[np.ndarray] | None = None,
    ) -> None:
        T = len(proprio_raw)
        if T == 0:
            return
        if len(actions_v_raw) != T or len(actions_raw) != T:
            raise ValueError(
                f"trajectory length mismatch: proprio={T} base={len(actions_v_raw)} "
                f"deployed={len(actions_raw)}"
            )
        self.n_episodes += 1
        for t in range(T):
            steps_to_end = T - 1 - t
            mc = (gamma**steps_to_end) * float(success)
            s = proprio_raw[t].astype(np.float32)
            a = actions_raw[t].astype(np.float32)
            av = actions_v_raw[t].astype(np.float32)
            if features_raw is not None and t < len(features_raw):
                h = np.asarray(features_raw[t], dtype=np.float16)
            else:
                h = np.zeros(FEATURE_DIM, dtype=np.float16)
            self.features.append(h)
            self.proprio.append(s)
            self.actions.append(a)
            self.actions_v.append(av)
            self.returns.append(mc)
            self.successes.append(float(success))
            self.rewards.append(float(success) if t == T - 1 else 0.0)
            self.dones.append(float(t == T - 1))
        overflow = len(self.proprio) - self.max_transitions
        if overflow > 0:
            self.features = self.features[overflow:]
            self.proprio = self.proprio[overflow:]
            self.actions = self.actions[overflow:]
            self.actions_v = self.actions_v[overflow:]
            self.returns = self.returns[overflow:]
            self.successes = self.successes[overflow:]
            self.rewards = self.rewards[overflow:]
            self.dones = self.dones[overflow:]

    def has_both_outcomes(self) -> bool:
        successes = np.asarray(self.successes, dtype=np.float32)
        return bool(np.any(successes > 0.5) and np.any(successes <= 0.5))

    def _normalized_actions(
        self,
        actions: list[np.ndarray],
        idx: np.ndarray,
    ) -> np.ndarray:
        action_mean = self.model.action_mean.detach().cpu().numpy()
        action_std = self.model.action_std.detach().cpu().numpy()
        raw = np.stack([actions[i] for i in idx]).astype(np.float32)
        return ((raw - action_mean) / np.clip(action_std, 1e-3, None)).astype(np.float32)

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        n = len(self.proprio)
        if n == 0:
            raise RuntimeError("online replay empty")
        succ = np.asarray(self.successes) > 0.5
        pos_idx = np.where(succ)[0]
        neg_idx = np.where(~succ)[0]
        if len(pos_idx) == 0:
            pos_idx = np.arange(n)
        if len(neg_idx) == 0:
            neg_idx = np.arange(n)
        n_pos = max(1, int(round(batch_size * self.pos_frac)))
        n_neg = batch_size - n_pos
        idx = np.concatenate(
            [
                self.rng.choice(pos_idx, size=n_pos, replace=True),
                self.rng.choice(neg_idx, size=n_neg, replace=True),
            ]
        )
        self.rng.shuffle(idx)
        return {
            "features": torch.from_numpy(
                np.stack([self.features[i] for i in idx]).astype(np.float32)
            ),
            "proprio": torch.from_numpy(np.stack([self.proprio[i] for i in idx])),
            # Critic learns the action that produced the return.
            "actions": torch.from_numpy(self._normalized_actions(self.actions, idx)),
            # G is conditioned on the frozen MolmoAct2 action.
            "base_actions": torch.from_numpy(self._normalized_actions(self.actions_v, idx)),
            "returns": torch.from_numpy(np.asarray([self.returns[i] for i in idx], dtype=np.float32)),
            "successes": torch.from_numpy(
                np.asarray([self.successes[i] for i in idx], dtype=np.float32)
            ),
            "rewards": torch.from_numpy(np.asarray([self.rewards[i] for i in idx], dtype=np.float32)),
            "dones": torch.from_numpy(np.asarray([self.dones[i] for i in idx], dtype=np.float32)),
        }

    def save_npz(self, path: Path, *, fit_norm_stats: bool = False) -> dict[str, float]:
        """Persist a VLA replay suitable for offline warmup.

        A G=0 collection can fit normalization to MolmoAct2 itself, avoiding
        the MolmoBot-demo/MolmoAct2 action-distribution mismatch.
        """
        if len(self) == 0:
            raise RuntimeError("cannot save an empty online replay")
        features = np.stack(self.features).astype(np.float16)
        proprio = np.stack(self.proprio).astype(np.float32)
        actions_raw = np.stack(self.actions).astype(np.float32)
        base_raw = np.stack(self.actions_v).astype(np.float32)

        if fit_norm_stats:
            proprio_mean = proprio.mean(axis=0, dtype=np.float64).astype(np.float32)
            proprio_std = proprio.std(axis=0, dtype=np.float64).clip(min=1e-3).astype(np.float32)
            action_mean = base_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
            action_std = base_raw.std(axis=0, dtype=np.float64).clip(min=1e-3).astype(np.float32)
            feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
            feature_std = features.std(axis=0, dtype=np.float64).clip(min=1e-3).astype(np.float32)
            self.model.set_norm_stats(
                proprio_mean,
                proprio_std,
                action_mean,
                action_std,
                feature_mean,
                feature_std,
            )
        else:
            proprio_mean = self.model.proprio_mean.detach().cpu().numpy()
            proprio_std = self.model.proprio_std.detach().cpu().numpy()
            action_mean = self.model.action_mean.detach().cpu().numpy()
            action_std = self.model.action_std.detach().cpu().numpy()
            feature_mean = self.model.feature_mean.detach().cpu().numpy()
            feature_std = self.model.feature_std.detach().cpu().numpy()

        action_std_safe = np.clip(action_std, 1e-3, None)
        arrays = {
            "features": features,
            "proprio": proprio,
            "actions": ((actions_raw - action_mean) / action_std_safe).astype(np.float32),
            "base_actions": ((base_raw - action_mean) / action_std_safe).astype(np.float32),
            "actions_raw": actions_raw,
            "base_actions_raw": base_raw,
            "returns": np.asarray(self.returns, dtype=np.float32),
            "successes": np.asarray(self.successes, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "proprio_mean": np.asarray(proprio_mean, dtype=np.float32),
            "proprio_std": np.asarray(proprio_std, dtype=np.float32),
            "action_mean": np.asarray(action_mean, dtype=np.float32),
            "action_std": np.asarray(action_std, dtype=np.float32),
            "feature_mean": np.asarray(feature_mean, dtype=np.float32),
            "feature_std": np.asarray(feature_std, dtype=np.float32),
            "n_transitions": np.asarray([len(self)], dtype=np.int64),
            "n_episodes": np.asarray([self.n_episodes], dtype=np.int64),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return {
            "transitions": float(len(self)),
            "episodes": float(self.n_episodes),
            "success_fraction": float(np.mean(self.successes)),
        }


def _make_cf_train_policy(cf_model: MolmoAct2CF, device: torch.device):
    """Build MolmoSpaces policy class that applies CF residual client-side."""
    from molmo_spaces.policy.learned_policy.molmoact2_policy import MolmoAct2_Policy

    class CFTrainPolicy(MolmoAct2_Policy):
        def __init__(self, exp_config):
            super().__init__(exp_config)
            self.cf_model = cf_model
            self.device = device
            self.ep_proprio: list[np.ndarray] = []
            self.ep_features: list[np.ndarray] = []
            self.ep_actions_v: list[np.ndarray] = []
            self.ep_actions: list[np.ndarray] = []
            self.ep_deltas: list[np.ndarray] = []
            self.ep_feature_ages: list[int] = []
            self._chunk_a_v: list[np.ndarray] | None = None
            self._chunk_feat: np.ndarray | None = None
            self.last_residual_rms = 0.0
            self.enable_g = True
            # Normalized-space residual exploration while learning (eval keeps 0).
            self.explore_residual_std = 0.0

        def reset(self):
            super().reset()
            self.ep_proprio = []
            self.ep_features = []
            self.ep_actions_v = []
            self.ep_actions = []
            self.ep_deltas = []
            self.ep_feature_ages = []
            self._chunk_a_v = None
            self._chunk_feat = None
            self.last_residual_rms = 0.0

        def inference_model(self, model_input):
            if self.actions_buffer is None or self.current_buffer_index >= min(
                self.chunk_size, len(self.actions_buffer or [])
            ):
                try:
                    import json_numpy

                    json_numpy.patch()
                except ImportError:
                    pass
                payload = {
                    "external_cam": np.asarray(model_input["external_cam"], dtype=np.uint8),
                    "wrist_cam": np.asarray(model_input["wrist_cam"], dtype=np.uint8),
                    "instruction": model_input["instruction"],
                    "state": np.asarray(model_input["state"], dtype=np.float32),
                    "timestamp": model_input.get("timestamp", time.time()),
                }
                r = self.session.post(self.url, json=payload, timeout=120)
                r.raise_for_status()
                resp = r.json()
                actions_v = np.asarray(resp["actions"], dtype=np.float32)
                if actions_v.ndim == 1:
                    actions_v = actions_v.reshape(1, -1)
                if actions_v.ndim != 2 or actions_v.shape[1] != 8:
                    raise ValueError(f"invalid expert action shape {actions_v.shape}")
                if not np.isfinite(actions_v).all():
                    raise FloatingPointError("expert returned non-finite actions")
                if "features" in resp:
                    feat = np.asarray(resp["features"], dtype=np.float32).reshape(-1)
                else:
                    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
                if feat.shape != (FEATURE_DIM,) or not np.isfinite(feat).all():
                    raise FloatingPointError(
                        f"invalid VLA features shape={feat.shape} finite={np.isfinite(feat).all()}"
                    )
                self._chunk_feat = feat
                # Keep the frozen action chunk intact. G is applied to each row
                # when that row is actually executed, using matching current
                # proprio instead of the chunk-start proprio for every row.
                self.actions_buffer = [row.copy() for row in actions_v]
                self._chunk_a_v = [row.copy() for row in actions_v]
                self.current_buffer_index = 0

            assert (
                self.actions_buffer is not None
                and self._chunk_a_v is not None
                and self._chunk_feat is not None
            )
            a_v = self._chunk_a_v[self.current_buffer_index]
            feature_age = self.current_buffer_index
            state = np.asarray(model_input["state"], dtype=np.float32)
            out = np.asarray(a_v, dtype=np.float32).copy()
            delta = np.zeros_like(out)
            if self.enable_g or self.explore_residual_std > 0.0:
                s = torch.from_numpy(np.array(state, dtype=np.float32, copy=True)).unsqueeze(0).to(
                    self.device
                )
                h = torch.from_numpy(
                    np.array(self._chunk_feat, dtype=np.float32, copy=True)
                ).unsqueeze(0).to(self.device)
                a = torch.from_numpy(np.array(a_v, dtype=np.float32, copy=True)).unsqueeze(0).to(
                    self.device
                )
                with torch.no_grad():
                    if self.enable_g:
                        refined, residual = self.cf_model.refine_raw(
                            s,
                            a,
                            features=h,
                            delta_clip=self.cf_model.refiner.max_delta,
                        )
                        out = refined.squeeze(0).cpu().numpy()
                        delta = residual.squeeze(0).cpu().numpy()
                    if self.explore_residual_std > 0.0:
                        # Explore in normalized action space, then denormalize.
                        a_n = self.cf_model.normalize_action(a)
                        noise_n = torch.randn_like(a_n) * float(self.explore_residual_std)
                        max_d = float(self.cf_model.refiner.max_delta)
                        noise_n = torch.clamp(noise_n, -max_d, max_d)
                        noisy = self.cf_model.denormalize_action(a_n + noise_n)
                        # If G is on, add exploration on top of the refined action
                        # but keep the total residual within max_delta of a_v.
                        if self.enable_g:
                            a_n_ref = self.cf_model.normalize_action(
                                torch.from_numpy(out.astype(np.float32)).unsqueeze(0).to(self.device)
                            )
                            total_n = torch.clamp(
                                a_n_ref + noise_n - a_n, -max_d, max_d
                            )
                            out = self.cf_model.denormalize_action(a_n + total_n).squeeze(0).cpu().numpy()
                            delta = out - np.asarray(a_v, dtype=np.float32)
                        else:
                            out = noisy.squeeze(0).cpu().numpy()
                            delta = out - np.asarray(a_v, dtype=np.float32)
            self.current_buffer_index += 1
            self.ep_deltas.append(np.asarray(delta, dtype=np.float32).copy())
            self.last_residual_rms = float(
                np.sqrt(np.mean(np.square(np.stack(self.ep_deltas))))
            )
            self.ep_feature_ages.append(feature_age)
            self.ep_proprio.append(state.copy())
            self.ep_features.append(self._chunk_feat.copy())
            self.ep_actions_v.append(np.asarray(a_v, dtype=np.float32).copy())
            self.ep_actions.append(np.asarray(out, dtype=np.float32).copy())
            return out

        def pop_episode(self) -> dict[str, Any]:
            data = {
                "states": self.ep_proprio,  # back-compat alias = proprio
                "proprio": self.ep_proprio,
                "features": self.ep_features,
                "actions_v": self.ep_actions_v,
                "actions": self.ep_actions,
                "residual_rms": self.last_residual_rms,
                "feature_age_mean": (
                    float(np.mean(self.ep_feature_ages)) if self.ep_feature_ages else 0.0
                ),
                "feature_age_max": max(self.ep_feature_ages, default=0),
            }
            self.ep_proprio = []
            self.ep_features = []
            self.ep_actions_v = []
            self.ep_actions = []
            self.ep_deltas = []
            self.ep_feature_ages = []
            return data

    return CFTrainPolicy


def _default_bench() -> Path:
    candidates = [
        Path(os.path.expanduser("~/.cache/molmospaces/assets"))
        / "benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench"
        / "FrankaPickDroidMiniBench_json_benchmark_20251231",
        Path(os.path.expanduser("~/.cache/molmo-spaces-resources"))
        / "benchmarks/molmospaces-bench-v1/20260408/procthor-10k/FrankaPickDroidMiniBench"
        / "FrankaPickDroidMiniBench_json_benchmark_20251231",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError("FrankaPickDroidMiniBench not found under HF caches")


def _mix_batch(
    offline: StratifiedReplay | None,
    online: OnlineReplay,
    batch_size: int,
    online_frac: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n_on = max(1, int(round(batch_size * online_frac))) if len(online) > 0 else 0
    n_off = batch_size - n_on
    parts = []
    if n_on > 0:
        parts.append(online.sample_batch(n_on))
    if n_off > 0 and offline is not None:
        parts.append(offline.sample_batch(n_off))
    if not parts:
        raise RuntimeError("no data to sample")
    for part in parts:
        if "base_actions" not in part and "actions" in part:
            # Offline expert data has G=0, so deployed == base.
            part["base_actions"] = part["actions"]
    if len(parts) == 1:
        return {k: v.to(device) for k, v in parts[0].items()}
    # Intersection of keys so legacy offline + feature online can mix safely.
    keys = set(parts[0].keys())
    for p in parts[1:]:
        keys &= set(p.keys())
    if not keys:
        raise RuntimeError("batch parts have no shared keys")
    return {k: torch.cat([p[k] for p in parts], dim=0).to(device) for k in keys}


def train_full(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    # --- Load / init CF ---
    if args.cf_ckpt and Path(args.cf_ckpt).is_file():
        model = MolmoAct2CF.load(args.cf_ckpt, map_location=device).to(device)
        log.info("Loaded CF ckpt %s", args.cf_ckpt)
    else:
        model = MolmoAct2CF(use_vla_features=True).to(device)
        log.info("Initialized fresh CF modules (VLA features)")
    if not model.bounded_critic and (
        args.updates_per_episode > 0 or args.phase1_steps > 0 or args.phase2_steps > 0
    ):
        raise ValueError(
            "Refusing to train a legacy unbounded-Q checkpoint. Collect/evaluate with "
            "--updates_per_episode 0, or start a fresh bounded critic."
        )

    offline = None
    if args.buffer and Path(args.buffer).is_file():
        arrays = load_buffer(Path(args.buffer))
        compatible = "features" in arrays or not model.use_vla_features
        if compatible and "proprio_mean" in arrays:
            model.set_norm_stats(
                arrays["proprio_mean"],
                arrays["proprio_std"],
                arrays["action_mean"],
                arrays["action_std"],
                arrays.get("feature_mean"),
                arrays.get("feature_std"),
            )
        elif compatible and "state_mean" in arrays:
            model.set_norm_stats(
                arrays["state_mean"],
                arrays["state_std"],
                arrays["action_mean"],
                arrays["action_std"],
            )
        if compatible:
            offline = StratifiedReplay(arrays, pos_frac=args.pos_frac, seed=args.seed)
            log.info("Offline buffer n=%d", len(offline))
        else:
            log.warning(
                "Ignoring proprio-only buffer and its normalization for VLA training; "
                "collect a matched G=0 VLA replay first"
            )

    online = OnlineReplay(model, max_transitions=args.online_capacity, pos_frac=args.pos_frac)

    opt_q, opt_g, opt_alpha = build_optimizers(
        model,
        lr_q=args.lr_q,
        lr_g=args.lr_g,
        lr_alpha=args.lr_alpha,
    )

    # Optional short offline refresh before online.
    if args.phase1_steps > 0 and offline is not None:
        log.info("[warmup] phase1 steps=%d", args.phase1_steps)
        for step in range(1, args.phase1_steps + 1):
            batch = {k: v.to(device) for k, v in offline.sample_batch(args.batch_size).items()}
            critic_step(
                model,
                batch,
                opt_q,
                cql_coef=args.cql_coef,
                cql_n_actions=args.cql_n_actions,
                cql_action_radius=args.cql_action_radius,
                cql_margin=args.cql_margin,
                cql_far_scale=args.cql_far_scale,
            )
    if args.phase2_steps > 0 and offline is not None:
        log.info("[warmup] phase2 steps=%d", args.phase2_steps)
        for step in range(1, args.phase2_steps + 1):
            batch = {k: v.to(device) for k, v in offline.sample_batch(args.batch_size).items()}
            if step % 2 == 0:
                critic_step(
                    model,
                    batch,
                    opt_q,
                    cql_coef=args.cql_coef * 0.5,
                    cql_n_actions=args.cql_n_actions,
                    cql_action_radius=args.cql_action_radius,
                    cql_margin=args.cql_margin,
                    cql_far_scale=args.cql_far_scale,
                )
            refiner_step(
                model,
                batch,
                opt_g,
                opt_alpha,
                target_divergence=args.target_divergence,
            )

    # --- Online env loop ---
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.assets_dir:
        os.environ["MLSPACES_ASSETS_DIR"] = args.assets_dir

    from molmo_spaces.evaluation.eval_main import run_evaluation

    CFTrainPolicy = _make_cf_train_policy(model, device)
    # Build a disposable config only to construct the policy once.
    from molmo_spaces.evaluation.configs.evaluation_configs import MolmoAct2PolicyEvalConfig

    exp_cfg = MolmoAct2PolicyEvalConfig()
    # Point at this shard's MolmoAct2 server (one per GPU).
    exp_cfg.policy_config.remote_config = {
        "host": args.server_host,
        "port": int(args.server_port),
    }
    policy = CFTrainPolicy(exp_cfg)
    if args.policy_chunk_size > 0:
        policy.chunk_size = args.policy_chunk_size
    requested_g = not args.disable_g
    policy.enable_g = requested_g and (args.force_g or args.g_start_episodes <= 0)
    # Exploration only while learning (never during pure eval).
    policy.explore_residual_std = (
        float(args.explore_residual_std) if args.updates_per_episode > 0 else 0.0
    )

    bench = Path(args.benchmark_dir) if args.benchmark_dir else _default_bench()
    start_ep = int(args.start_episode)
    end_ep = start_ep + int(args.num_episodes)
    log.info(
        "Online train: episodes=[%d,%d) n=%d log_every=%d server=%s:%d bench=%s enable_g=%s",
        start_ep,
        end_ep,
        args.num_episodes,
        args.log_every_episodes,
        args.server_host,
        args.server_port,
        bench,
        policy.enable_g,
    )

    recent_success = deque(maxlen=args.log_every_episodes)
    all_success: list[float] = []
    t0 = time.time()
    last_q_info: dict[str, float] = {}
    last_g_info: dict[str, float] = {}
    local_ep = 0

    for ep in range(start_ep, end_ep):
        local_ep += 1
        recent_adv = float(last_g_info.get("predicted_advantage", 0.0))
        policy.enable_g = bool(
            requested_g
            and (
                args.force_g
                or (
                    len(all_success) >= args.g_start_episodes
                    and online.has_both_outcomes()
                    and critic_is_healthy(last_q_info, args.g_max_mc_loss)
                    and recent_adv >= args.g_min_advantage
                )
            )
        )
        ep_t0 = time.time()
        ep_out = out_dir / "rollouts" / f"ep_{ep:05d}"
        ep_out.mkdir(parents=True, exist_ok=True)
        rollout_ok = True
        try:
            results = run_evaluation(
                eval_config_cls=MolmoAct2PolicyEvalConfig,
                benchmark_dir=bench,
                task_horizon_steps=args.horizon,
                num_workers=1,
                use_wandb=False,
                preloaded_policy=policy,
                episode_idx=ep,
                output_dir=ep_out,
            )
            success = bool(results.success_count > 0)
        except Exception as e:  # noqa: BLE001
            log.exception("episode %d failed: %s", ep, e)
            success = False
            rollout_ok = False
        traj = policy.pop_episode()
        if not rollout_ok or not traj["states"]:
            log.warning(
                "dropping invalid/partial episode %d rollout_ok=%s steps=%d",
                ep,
                rollout_ok,
                len(traj["states"]),
            )
            continue
        online.add_episode(
            traj.get("proprio", traj["states"]),
            traj["actions_v"],
            traj["actions"],
            success=success,
            gamma=args.gamma,
            features_raw=traj.get("features"),
        )
        recent_success.append(float(success))
        all_success.append(float(success))

        # Gradient updates after each env episode.
        if len(online) >= args.batch_size // 4 or (offline is not None and len(offline) > 0):
            for update_idx in range(args.updates_per_episode):
                batch = _mix_batch(
                    offline,
                    online,
                    args.batch_size,
                    online_frac=args.online_frac,
                    device=device,
                )
                last_q_info = critic_step(
                    model,
                    batch,
                    opt_q,
                    cql_coef=args.cql_coef * 0.5,
                    cql_n_actions=args.cql_n_actions,
                    cql_action_radius=args.cql_action_radius,
                    cql_margin=args.cql_margin,
                    cql_far_scale=args.cql_far_scale,
                )
                can_update_g = (
                    requested_g
                    and len(all_success) >= args.g_start_episodes
                    and online.has_both_outcomes()
                    and critic_is_healthy(last_q_info, args.g_max_mc_loss)
                )
                if can_update_g and update_idx % args.policy_delay == 0:
                    last_g_info = refiner_step(
                        model,
                        batch,
                        opt_g,
                        opt_alpha,
                        target_divergence=args.target_divergence,
                    )

        ep_dt = time.time() - ep_t0
        log.info(
            "ep %d/%d (global %d) success=%s steps=%d g=%s residual_rms=%.4f "
            "feature_age=%.2f dt=%.1fs online_n=%d",
            local_ep,
            args.num_episodes,
            ep,
            success,
            len(traj["states"]),
            policy.enable_g,
            float(traj.get("residual_rms", 0.0)),
            float(traj.get("feature_age_mean", 0.0)),
            ep_dt,
            len(online),
        )

        # Log every N environment episodes (a "batch" of env episodes).
        if local_ep % args.log_every_episodes == 0 or local_ep == args.num_episodes:
            window = list(recent_success)
            row = {
                "shard_start": start_ep,
                "shard_end": end_ep,
                "global_episode": ep,
                "local_episodes": local_ep,
                "batch": local_ep // args.log_every_episodes,
                "window_episodes": len(window),
                "window_success_rate": float(np.mean(window)) if window else 0.0,
                "cumulative_success_rate": float(np.mean(all_success)),
                "online_transitions": len(online),
                "g_enabled": policy.enable_g,
                "residual_rms_last": float(traj.get("residual_rms", 0.0)),
                "feature_age_mean": float(traj.get("feature_age_mean", 0.0)),
                "elapsed_sec": time.time() - t0,
                "server_port": int(args.server_port),
                **{f"q_{k}": v for k, v in last_q_info.items()},
                **{f"g_{k}": v for k, v in last_g_info.items()},
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            ckpt = out_dir / f"molmoact2_cf_lep{local_ep:05d}_gep{ep:05d}.pt"
            model.save(
                str(ckpt),
                meta={"local_episodes": local_ep, "global_episode": ep, "metrics": row},
            )
            model.save(str(out_dir / "molmoact2_cf_latest.pt"), meta=row)
            if args.replay_out:
                online.save_npz(
                    Path(args.replay_out),
                    fit_norm_stats=args.fit_replay_norm_stats,
                )
            log.info(
                "METRICS batch=%d local_eps=%d global_ep=%d window_sr=%.3f cum_sr=%.3f "
                "adv=%.4f rms=%.4f online_n=%d ckpt=%s",
                row["batch"],
                row["local_episodes"],
                row["global_episode"],
                row["window_success_rate"],
                row["cumulative_success_rate"],
                last_g_info.get("predicted_advantage", float("nan")),
                last_g_info.get("residual_rms", float("nan")),
                len(online),
                ckpt.name,
            )

    summary = {
        "start_episode": start_ep,
        "num_episodes": args.num_episodes,
        "server_port": int(args.server_port),
        "cumulative_success_rate": float(np.mean(all_success)) if all_success else 0.0,
        "elapsed_sec": time.time() - t0,
        "metrics_path": str(metrics_path),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if args.replay_out and len(online) > 0:
        replay_info = online.save_npz(
            Path(args.replay_out),
            fit_norm_stats=args.fit_replay_norm_stats,
        )
        log.info("Saved replay %s: %s", args.replay_out, replay_info)
    log.info("Done: %s", json.dumps(summary))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cf_ckpt", type=str, default=str(_HERE / "runs/molmoact2_cf_smoke/molmoact2_cf.pt"))
    p.add_argument("--buffer", type=str, default=str(_HERE / "runs/pick_buffer.npz"))
    p.add_argument("--out_dir", type=str, default=str(_HERE / "runs/molmoact2_cf_full"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--benchmark_dir", type=str, default="")
    p.add_argument("--assets_dir", type=str, default=os.path.expanduser("~/.cache/molmospaces/assets"))
    p.add_argument("--start_episode", type=int, default=0, help="First MiniBench episode index")
    p.add_argument("--num_episodes", type=int, default=1000, help="Episodes in this shard")
    p.add_argument("--server_host", type=str, default="localhost")
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument(
        "--log_every_episodes",
        type=int,
        default=100,
        help="Log metrics every this many environment episodes",
    )
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--updates_per_episode", type=int, default=5)
    p.add_argument("--online_frac", type=float, default=0.5)
    p.add_argument("--online_capacity", type=int, default=200_000)
    p.add_argument("--phase1_steps", type=int, default=0, help="Optional offline refresh before online")
    p.add_argument("--phase2_steps", type=int, default=0)
    p.add_argument("--disable_g", action="store_true")
    p.add_argument(
        "--force_g",
        action="store_true",
        help="Enable a frozen checkpoint G without critic gating (evaluation only)",
    )
    p.add_argument("--g_start_episodes", type=int, default=20)
    p.add_argument("--g_max_mc_loss", type=float, default=0.20)
    p.add_argument(
        "--g_min_advantage",
        type=float,
        default=0.005,
        help="Deploy G only when recent predicted advantage clears this bar",
    )
    p.add_argument("--policy_delay", type=int, default=2)
    p.add_argument(
        "--policy_chunk_size",
        type=int,
        default=0,
        help="Override MolmoAct2 chunk size; 0 keeps the policy default",
    )
    p.add_argument(
        "--explore_residual_std",
        type=float,
        default=0.02,
        help="Normalized residual exploration noise while learning (0 for eval)",
    )
    p.add_argument("--replay_out", type=str, default="")
    p.add_argument("--fit_replay_norm_stats", action="store_true")
    p.add_argument("--lr_q", type=float, default=3e-4)
    p.add_argument("--lr_g", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--cql_coef", type=float, default=0.1)
    p.add_argument("--cql_n_actions", type=int, default=8)
    p.add_argument("--cql_action_radius", type=float, default=0.05)
    p.add_argument("--cql_far_scale", type=float, default=1.0)
    p.add_argument("--cql_margin", type=float, default=0.0)
    p.add_argument("--target_divergence", type=float, default=2.5e-3)
    p.add_argument("--pos_frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.policy_delay < 1:
        p.error("--policy_delay must be >= 1")
    if args.force_g and args.updates_per_episode != 0:
        p.error("--force_g is evaluation-only; use --updates_per_episode 0")
    if args.fit_replay_norm_stats and (not args.disable_g or args.updates_per_episode != 0):
        p.error(
            "--fit_replay_norm_stats is collection-only; use --disable_g "
            "--updates_per_episode 0"
        )
    return args


if __name__ == "__main__":
    train_full(parse_args())
