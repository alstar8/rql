"""V19 harness: iterated CFGRL extractor on frozen MolmoAct2 (flow-only).

Train split only: house0_kettle_v13/train (24 poses). The 12-episode val
split is never used for training or validation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import v13_harness as v13
from v13_harness import VariantSpec


RUN_NAME = "rlt_cf_v19_kettle"
BENCHMARK_NAME = v13.BENCHMARK_NAME
TRAIN_EPISODES = v13.TRAIN_EPISODES  # 24 train poses
VALIDATION_EPISODES = v13.VALIDATION_EPISODES  # 12; DO NOT TOUCH
HORIZON = v13.HORIZON
TRAIN_SEED = 20260817
N_CRITICS = v13.N_CRITICS
FLOW_STEPS = v13.FLOW_STEPS

PHASE_ROUNDS = int(os.environ.get("V19_PHASE_ROUNDS", "100"))
PHASE_B_EPISODES = int(os.environ.get("V19_PHASE_B_EPISODES", "1"))
PHASE_PROBE = os.environ.get("V19_PHASE_PROBE", "1") != "0"
PHASE_BARRIER = os.environ.get("V19_PHASE_BARRIER", "1") != "0"
PHASE_BARRIER_POLL_SEC = float(os.environ.get("V19_PHASE_BARRIER_POLL_SEC", "5"))
PHASE_BARRIER_LOG_SEC = float(os.environ.get("V19_PHASE_BARRIER_LOG_SEC", "60"))
MAX_VALID_EPISODES = int(
    os.environ.get("V19_MAX_VALID_EPISODES", str(PHASE_ROUNDS * PHASE_B_EPISODES))
)
TARGET_ENV_STEPS = int(
    os.environ.get(
        "V19_TARGET_ENV_STEPS",
        str(MAX_VALID_EPISODES * HORIZON + HORIZON),
    )
)
SNAPSHOT_EPISODES = tuple(
    int(token.strip())
    for token in os.environ.get(
        "V19_SNAPSHOT_EPISODES",
        f"0,{MAX_VALID_EPISODES}",
    ).split(",")
    if token.strip()
)
LOG_EVERY_EPISODES = int(os.environ.get("V19_LOG_EVERY_EPISODES", "1"))
CKPT_EVERY_EPISODES = int(
    os.environ.get("V19_CKPT_EVERY_EPISODES", str(MAX_VALID_EPISODES))
)
MAX_UPDATE_SEC_PER_EPISODE = float(
    os.environ.get("V19_MAX_UPDATE_SEC_PER_EPISODE", "45")
)
UPDATES_PER_EPISODE = int(os.environ.get("V19_UPDATES_PER_EPISODE", "16"))
BENCHMARK_POSE_CYCLE = max(
    1, int(os.environ.get("V19_POSE_CYCLE", str(TRAIN_EPISODES)))
)

ACTOR_MIXTURE_PROB_PRE_GATE = float(os.environ.get("V19_MIX_PRE_GATE", "0.0"))
ACTOR_MIXTURE_PROB_POST_GATE = float(os.environ.get("V19_MIX_POST_GATE", "0.25"))
ACTOR_MIXTURE_PROB_STRONG = float(os.environ.get("V19_MIX_STRONG", "0.5"))
ACTOR_MIXTURE_PROB_VERY = float(os.environ.get("V19_MIX_VERY", "0.75"))
POS_FRAC = float(os.environ.get("V19_POS_FRAC", "0.5"))
EMPIRICAL_MIN_EPISODES = int(os.environ.get("V19_EMPIRICAL_MIN_EPISODES", "16"))
CFGRL_W = float(os.environ.get("V19_CFGRL_W", "0.0"))
CFGRL_W_AFTER = float(os.environ.get("V19_CFGRL_W_AFTER", "0.5"))
CFGRL_UNCOND_MSE_MAX = float(os.environ.get("V19_CFGRL_UNCOND_MSE_MAX", "0.02"))
CFGRL_KQ = int(os.environ.get("V19_CFGRL_KQ", "4096"))
CFGRL_KPI = int(os.environ.get("V19_CFGRL_KPI", "2048"))
CFGRL_KQ_ONLINE = int(os.environ.get("V19_CFGRL_KQ_ONLINE", "2048"))
CFGRL_KPI_ONLINE = int(os.environ.get("V19_CFGRL_KPI_ONLINE", "1024"))
CFGRL_O_DIM = int(os.environ.get("V19_CFGRL_O_DIM", "128"))
HIDDEN = int(os.environ.get("V19_HIDDEN", "1024"))
N_HIDDEN_ACTOR = int(os.environ.get("V19_N_HIDDEN_ACTOR", "5"))
N_HIDDEN_CRITIC = int(os.environ.get("V19_N_HIDDEN_CRITIC", "4"))
Z_EXPAND_DIM = int(os.environ.get("V19_Z_EXPAND_DIM", "512"))
LAYERNORM_HEADS = os.environ.get("V19_LAYERNORM_HEADS", "1") != "0"
CFGRL_ROUND = int(os.environ.get("V19_CFGRL_ROUND", str(max(1, PHASE_B_EPISODES))))
SEED_REPLAY = os.environ.get("V19_SEED_REPLAY", "")
N_SHARDS = int(os.environ.get("V19_N_SHARDS", "32"))
REPLAY_CAPACITY = int(
    os.environ.get(
        "V19_REPLAY_CAPACITY",
        str(50000 + PHASE_ROUNDS * PHASE_B_EPISODES * HORIZON),
    )
)

HTTP_PORTS = tuple(range(8760, 8760 + max(1, min(N_SHARDS, 8))))

# SIGABRT/SIGSEGV from NVIDIA EGL `mjr_readPixels` (child return codes).
EGL_CRASH_RETURNCODES = frozenset({-11, -6, 6, 134, 139})


def isolated_rollouts_enabled() -> bool:
    raw = os.environ.get("RLT_ISOLATED_ROLLOUT", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def isolated_rollout_attempts() -> int:
    return max(1, int(os.environ.get("RLT_ISOLATED_ATTEMPTS", "4")))


def isolated_rollout_timeout_sec(horizon: int) -> float:
    """Wall budget for one healthy isolated episode, including scene load.

    This host finishes a 500-step probe in ~2–4 min. The old 45 min cap let a
    silent import hang block the 32-shard phase barrier. Override with
    ``RLT_ISOLATED_TIMEOUT_SEC``.
    """

    override = os.environ.get("RLT_ISOLATED_TIMEOUT_SEC", "").strip()
    if override:
        return float(max(60.0, float(override)))
    return float(min(900.0, max(480.0, int(horizon) * 1.5 + 180)))


def isolated_rollout_startup_sec() -> float:
    """Kill a child that never reaches isolated main (import/init hang)."""

    override = os.environ.get("RLT_ISOLATED_STARTUP_SEC", "").strip()
    if override:
        return float(max(15.0, float(override)))
    return 180.0


def is_egl_crash_returncode(code: int | None) -> bool:
    if code is None:
        return True
    return int(code) in EGL_CRASH_RETURNCODES

VARIANTS = (
    VariantSpec(
        "flow_cfgrl",
        0,
        "flow",
        "rlt",
        False,
        False,
        "flow",
        UPDATES_PER_EPISODE,
        8760,
        "FlowVelocityActor",
        "EnsembleTimeCQL",
        None,
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def phase_schedule(n_rounds: int = PHASE_ROUNDS) -> tuple[str, ...]:
    """Ordered phase labels: 0, 1A, 1B, 2A, 2B, ..."""

    if n_rounds < 0:
        raise ValueError(f"n_rounds must be >= 0, got {n_rounds}")
    labels = ["0"]
    for round_idx in range(1, n_rounds + 1):
        labels.append(f"{round_idx}A")
        labels.append(f"{round_idx}B")
    return tuple(labels)


def next_phase_label(phase: str, n_rounds: int = PHASE_ROUNDS) -> str | None:
    """Phase after ``phase`` in the CPI schedule, or None if it is the last."""

    labels = phase_schedule(n_rounds)
    try:
        idx = labels.index(str(phase))
    except ValueError as error:
        raise ValueError(f"unknown phase {phase!r}") from error
    if idx + 1 >= len(labels):
        return None
    return labels[idx + 1]


def phase_slot(phase: str) -> int:
    """Map 0 / rA / rB onto 0, 1, 2, 3, ..."""

    label = str(phase).strip()
    if label == "0":
        return 0
    if len(label) >= 2 and label[-1] in {"A", "B"}:
        round_idx = int(label[:-1])
        if round_idx < 1:
            raise ValueError(f"phase round must be >= 1, got {phase!r}")
        return 2 * round_idx - (1 if label[-1] == "A" else 0)
    raise ValueError(f"unknown phase {phase!r}")


def phase_probes_per_worker(n_poses: int, n_shards: int) -> int:
    """Copies per worker so the fleet covers every train pose at least once."""

    if n_poses <= 0:
        raise ValueError(f"n_poses must be >= 1, got {n_poses}")
    if n_shards <= 0:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    return max(1, (int(n_poses) + int(n_shards) - 1) // int(n_shards))


def phase_probe_episode_indices(
    shard: int,
    n_poses: int,
    n_shards: int = N_SHARDS,
) -> tuple[int, ...]:
    """Fixed poses per worker; the set does not rotate by phase.

    32 workers / 24 poses: one probe, pose s % 24 (workers 24–31 repeat 0–7).
    8 workers / 24 poses: three probes, poses s, s+8, s+16 (covers all 24).
    """

    n_copies = phase_probes_per_worker(n_poses, n_shards)
    shard_id = int(shard) if int(shard) >= 0 else 0
    return tuple(
        (copy * int(n_shards) + shard_id) % int(n_poses) for copy in range(n_copies)
    )


def phase_probe_episode_idx(
    shard: int,
    phase: str,
    n_poses: int,
    n_shards: int = N_SHARDS,
) -> int:
    """First fixed pose for this worker (kept for older call sites)."""

    del phase
    return phase_probe_episode_indices(shard, n_poses, n_shards=n_shards)[0]


def phase_probe_row_is_complete(rec: dict[str, Any]) -> bool:
    """True for a finished probe. SIGTERM junk is valid=false / n_steps=0."""

    if rec.get("valid") is False:
        return False
    if "n_steps" in rec and int(rec.get("n_steps", 0) or 0) <= 0:
        return False
    return True


def shard_phase_probe_count(shard_dir: Path, phase: str) -> int:
    target = str(phase)
    return sum(
        1
        for rec in _load_phase_probe_rows(shard_dir)
        if str(rec.get("phase", "")) == target
    )


def shard_has_phase(
    shard_dir: Path,
    phase: str,
    *,
    n_required: int = 1,
) -> bool:
    return shard_phase_probe_count(shard_dir, phase) >= max(1, int(n_required))


def last_recorded_phase(shard_dir: Path) -> str | None:
    """Highest phase_slot recorded in this shard's phase_probe.jsonl."""

    best: str | None = None
    best_slot = -1
    for rec in _load_phase_probe_rows(shard_dir):
        label = str(rec.get("phase", "")).strip()
        if not label:
            continue
        try:
            slot = phase_slot(label)
        except ValueError:
            continue
        if slot >= best_slot:
            best_slot = slot
            best = label
    return best


def phase_barrier_status(
    run_dir: Path,
    phase: str,
    *,
    n_shards: int = N_SHARDS,
    n_poses: int = TRAIN_EPISODES,
    variant: str = "flow_cfgrl",
) -> tuple[int, list[int]]:
    """How many shards have logged `phase`, and which shard ids are missing."""

    if n_shards <= 0:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    n_required = phase_probes_per_worker(n_poses, n_shards)
    run_dir = Path(run_dir)
    missing: list[int] = []
    for shard in range(n_shards):
        shard_dir = run_dir / variant / f"shard_{shard}"
        if not shard_has_phase(shard_dir, phase, n_required=n_required):
            missing.append(shard)
    return n_shards - len(missing), missing


def wait_for_phase_barrier(
    run_dir: Path,
    phase: str,
    *,
    n_shards: int = N_SHARDS,
    n_poses: int = TRAIN_EPISODES,
    variant: str = "flow_cfgrl",
    poll_sec: float = PHASE_BARRIER_POLL_SEC,
    log_every_sec: float = PHASE_BARRIER_LOG_SEC,
    should_stop: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> bool:
    """Block until all shards have recorded `phase`.

    Returns True when the barrier is satisfied, False if ``should_stop``.
    """

    poll = max(0.05, float(poll_sec))
    log_every = max(poll, float(log_every_sec))
    last_log = 0.0
    while True:
        n_ready, missing = phase_barrier_status(
            run_dir, phase, n_shards=n_shards, n_poses=n_poses, variant=variant
        )
        if not missing:
            if log is not None:
                log(f"CFGRL phase barrier phase={phase} complete {n_ready}/{n_shards}")
            return True
        if should_stop is not None and should_stop():
            if log is not None:
                log(f"CFGRL phase barrier phase={phase} interrupted")
            return False
        now = time.monotonic()
        if log is not None and now - last_log >= log_every:
            preview = ",".join(str(shard) for shard in missing[:12])
            extra = "" if len(missing) <= 12 else f"+{len(missing) - 12}"
            log(
                f"CFGRL phase barrier phase={phase} waiting "
                f"{n_ready}/{n_shards} missing=[{preview}{extra}]"
            )
            last_log = now
        time.sleep(poll)


def _load_phase_probe_rows(shard_dir: Path) -> list[dict[str, Any]]:
    path = shard_dir / "phase_probe.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and rec.get("phase") and phase_probe_row_is_complete(rec):
                rows.append(rec)
    return rows


def aggregate_phase_sr(
    run_dir: Path,
    *,
    n_shards: int = N_SHARDS,
    n_rounds: int = PHASE_ROUNDS,
    n_poses: int = TRAIN_EPISODES,
    variant: str = "flow_cfgrl",
) -> list[dict[str, Any]]:
    """Pool per-worker probes into a phase SR table."""

    run_dir = Path(run_dir)
    schedule = phase_schedule(n_rounds)
    n_expected = n_shards * phase_probes_per_worker(n_poses, n_shards)
    by_phase: dict[str, list[dict[str, Any]]] = {label: [] for label in schedule}
    for shard in range(n_shards):
        shard_dir = run_dir / variant / f"shard_{shard}"
        seen: set[tuple[int, str, int]] = set()
        for rec in _load_phase_probe_rows(shard_dir):
            phase = str(rec["phase"])
            episode_idx = int(rec.get("episode_idx", -1))
            key = (shard, phase, episode_idx)
            if phase not in by_phase or key in seen:
                continue
            seen.add(key)
            by_phase[phase].append(rec)
    table: list[dict[str, Any]] = []
    for phase in schedule:
        recs = by_phase[phase]
        n = len(recs)
        successes = sum(int(bool(rec.get("success"))) for rec in recs)
        complete = n >= n_expected
        table.append(
            {
                "phase": phase,
                "n": n,
                "n_expected": n_expected,
                "successes": successes,
                "sr": (successes / n) if n else None,
                "complete": complete,
                "pending_shards": max(0, n_expected - n),
                "policy": recs[0].get("policy") if recs else None,
            }
        )
    return table


def cfgrl_collect_mixture_prob(
    run_dir: Path,
    *,
    n_shards: int = N_SHARDS,
    n_rounds: int = PHASE_ROUNDS,
    n_poses: int = TRAIN_EPISODES,
    variant: str = "flow_cfgrl",
    pre: float = ACTOR_MIXTURE_PROB_PRE_GATE,
    post: float = ACTOR_MIXTURE_PROB_POST_GATE,
    strong: float = ACTOR_MIXTURE_PROB_STRONG,
    very: float = ACTOR_MIXTURE_PROB_VERY,
) -> float:
    """Actor collect p after Phase A if the latest complete rA probe is not worse than Phase 0.

    Phase 0 is the frozen-VLA baseline on the *same* pose set. Ignore rB (same
    actor, different noise). Allow 1/n noise so a true tie still mixes.
    Raise p as the extractor pulls ahead of the prior (0.25 / 0.5 / 0.75).
    """

    rows = aggregate_phase_sr(
        run_dir,
        n_shards=n_shards,
        n_rounds=n_rounds,
        n_poses=n_poses,
        variant=variant,
    )
    by_phase = {row["phase"]: row for row in rows}
    baseline = by_phase.get("0")
    if (
        baseline is None
        or not baseline["complete"]
        or not isinstance(baseline["sr"], float)
    ):
        return float(pre)
    latest = None
    for row in rows:
        phase = str(row["phase"])
        if not phase.endswith("A"):
            continue
        if row["complete"] and isinstance(row["sr"], float):
            latest = row
    if latest is None:
        return float(pre)
    n_expected = max(1, int(baseline.get("n_expected") or n_shards))
    gap = float(latest["sr"]) - float(baseline["sr"])
    if gap + 1.0 / float(n_expected) < 0.0:
        return float(pre)
    if gap >= 0.10:
        return float(very)
    if gap >= 0.03:
        return float(strong)
    return float(post)


def format_phase_sr_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# V19 phase SR (fixed pose set; no rotation by phase)",
        "",
        "| Phase | n | successes | SR | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        sr = row["sr"]
        sr_s = f"{100.0 * sr:.1f}%" if isinstance(sr, float) else "—"
        status = "complete" if row["complete"] else f"pending {row['pending_shards']}"
        lines.append(
            f"| {row['phase']} | {row['n']}/{row['n_expected']} | "
            f"{row['successes']} | {sr_s} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_phase_sr_table(
    run_dir: Path,
    *,
    n_shards: int = N_SHARDS,
    n_rounds: int = PHASE_ROUNDS,
    n_poses: int = TRAIN_EPISODES,
    variant: str = "flow_cfgrl",
) -> Path:
    run_dir = Path(run_dir)
    rows = aggregate_phase_sr(
        run_dir,
        n_shards=n_shards,
        n_rounds=n_rounds,
        n_poses=n_poses,
        variant=variant,
    )
    table_path = run_dir / "PHASE_SR.md"
    json_path = run_dir / "phase_sr.jsonl"
    table_path.write_text(format_phase_sr_table(rows), encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return table_path


def _assert_v19_run_dir(run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    if resolved.name != RUN_NAME:
        raise ValueError(
            f"V19 run directory basename must be {RUN_NAME!r}, got {resolved}"
        )
    if "val" in {part.lower() for part in resolved.parts}:
        raise ValueError(f"V19 refuses val paths: {resolved}")
    return resolved


def _assert_train_benchmark(benchmark_train: Path) -> Path:
    resolved = benchmark_train.resolve()
    parts = {part.lower() for part in resolved.parts}
    if "val" in parts or resolved.name.lower() in {"val", "validation"}:
        raise ValueError(
            "V19 must not touch the 12-episode val split; "
            f"got benchmark_dir={resolved}"
        )
    if resolved.name != "train":
        raise ValueError(
            f"V19 benchmark_dir must be the train split (.../train), got {resolved}"
        )
    return resolved


def _append_option(command: list[str], flag: str, value: object | None = None) -> None:
    if flag in command:
        return
    command.append(flag)
    if value is not None:
        command.append(str(value))


def _set_argument(command: list[str], flag: str, value: object) -> None:
    index = command.index(flag)
    command[index + 1] = str(value)


def _apply_v19_flags(command: list[str], variant: VariantSpec) -> list[str]:
    _set_argument(command, "--max_valid_episodes", MAX_VALID_EPISODES)
    _set_argument(command, "--target_env_steps", TARGET_ENV_STEPS)
    _set_argument(
        command,
        "--snapshot_episodes",
        ",".join(str(episode) for episode in SNAPSHOT_EPISODES),
    )
    _set_argument(command, "--log_every_episodes", LOG_EVERY_EPISODES)
    _set_argument(command, "--ckpt_every_episodes", CKPT_EVERY_EPISODES)
    _set_argument(command, "--updates_per_episode", variant.updates_per_episode)
    if "--replay_capacity" in command:
        _set_argument(command, "--replay_capacity", REPLAY_CAPACITY)
    else:
        _append_option(command, "--replay_capacity", REPLAY_CAPACITY)
    _set_argument(command, "--ref_dropout", 0.5)
    if "--max_updates_per_episode" in command:
        _set_argument(command, "--max_updates_per_episode", variant.updates_per_episode)
    else:
        _append_option(command, "--max_updates_per_episode", variant.updates_per_episode)
    if "--max_update_sec_per_episode" in command:
        _set_argument(
            command,
            "--max_update_sec_per_episode",
            MAX_UPDATE_SEC_PER_EPISODE,
        )
    else:
        _append_option(
            command,
            "--max_update_sec_per_episode",
            MAX_UPDATE_SEC_PER_EPISODE,
        )
    _append_option(command, "--no_critic_target_use_guide")
    if "--shard_size" in command:
        _set_argument(command, "--shard_size", BENCHMARK_POSE_CYCLE)
    else:
        _append_option(command, "--shard_size", BENCHMARK_POSE_CYCLE)
    if "--start_episode" in command:
        _set_argument(command, "--start_episode", 0)
    else:
        _append_option(command, "--start_episode", 0)
    _append_option(command, "--benchmark_pose_cycle", BENCHMARK_POSE_CYCLE)
    while "--use_cf_guide" in command:
        command.remove("--use_cf_guide")
    _append_option(command, "--no_cf_guide")
    _append_option(command, "--cfgrl")
    _append_option(command, "--cfgrl_w", CFGRL_W)
    _append_option(command, "--cfgrl_w_after_warmup", CFGRL_W_AFTER)
    _append_option(command, "--cfgrl_uncond_mse_max", CFGRL_UNCOND_MSE_MAX)
    _append_option(command, "--cfgrl_kq", CFGRL_KQ)
    _append_option(command, "--cfgrl_kpi", CFGRL_KPI)
    _append_option(command, "--cfgrl_kq_online", CFGRL_KQ_ONLINE)
    _append_option(command, "--cfgrl_kpi_online", CFGRL_KPI_ONLINE)
    _append_option(command, "--cfgrl_o_dim", CFGRL_O_DIM)
    _append_option(command, "--hidden", HIDDEN)
    _append_option(command, "--n_hidden_actor", N_HIDDEN_ACTOR)
    _append_option(command, "--n_hidden_critic", N_HIDDEN_CRITIC)
    _append_option(command, "--z_expand_dim", Z_EXPAND_DIM)
    if LAYERNORM_HEADS:
        _append_option(command, "--layernorm_heads")
    _append_option(command, "--cfgrl_round_episodes", CFGRL_ROUND)
    _append_option(command, "--cfgrl_phase_rounds", PHASE_ROUNDS)
    _append_option(command, "--cfgrl_phase_b_episodes", PHASE_B_EPISODES)
    if PHASE_PROBE:
        _append_option(command, "--cfgrl_phase_probe")
    if PHASE_BARRIER:
        _append_option(command, "--cfgrl_phase_barrier")
        _append_option(command, "--cfgrl_n_shards", N_SHARDS)
    else:
        _append_option(command, "--no_cfgrl_phase_barrier")
    seed_replay = os.environ.get("V19_SEED_REPLAY", SEED_REPLAY)
    if seed_replay:
        _append_option(command, "--seed_replay", seed_replay)
    _append_option(command, "--actor_mixture_prob", ACTOR_MIXTURE_PROB_PRE_GATE)
    _append_option(command, "--actor_mixture_prob_pre_gate", ACTOR_MIXTURE_PROB_PRE_GATE)
    _append_option(
        command,
        "--actor_mixture_prob_post_gate",
        ACTOR_MIXTURE_PROB_POST_GATE,
    )
    if "--pos_frac" in command:
        _set_argument(command, "--pos_frac", POS_FRAC)
    else:
        _append_option(command, "--pos_frac", POS_FRAC)
    while "--always_collect_actor" in command:
        command.remove("--always_collect_actor")
    if "--require_empirical_gate" not in command:
        command.append("--require_empirical_gate")
    _append_option(command, "--g_min_empirical_advantage", 0.0)
    _append_option(command, "--empirical_min_episodes", EMPIRICAL_MIN_EPISODES)
    _append_option(command, "--freeze_token")
    return command


def build_train_command(
    *,
    variant: VariantSpec | str,
    root: Path,
    run_dir: Path,
    benchmark_train: Path,
    residual_checkpoint: Path,
    flow_checkpoint: Path,
    python_executable: str,
    tmp_rollout_dir: Path,
    fresh: bool = False,
) -> list[str]:
    if isinstance(variant, str):
        variant = VARIANT_BY_NAME[variant]
    run_dir = _assert_v19_run_dir(run_dir)
    benchmark_train = _assert_train_benchmark(benchmark_train)
    command = v13.build_train_command(
        python_executable=python_executable,
        root=root,
        run_dir=run_dir,
        benchmark_train=benchmark_train,
        residual_checkpoint=residual_checkpoint,
        flow_checkpoint=flow_checkpoint,
        tmp_rollout_dir=tmp_rollout_dir,
        variant=variant,
        fresh=fresh,
    )
    _set_argument(command, "--out_dir", str(run_dir / variant.name))
    _set_argument(command, "--seed", TRAIN_SEED)
    return _apply_v19_flags(command, variant)


def build_server_command(
    *,
    variant: VariantSpec | str,
    root: Path,
    checkpoint: Path,
    serve_prefix: Sequence[str],
) -> list[str]:
    if isinstance(variant, str):
        variant = VARIANT_BY_NAME[variant]
    return v13.build_server_command(
        serve_prefix=serve_prefix,
        root=root,
        variant=variant,
        checkpoint=checkpoint,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train-command")
    train.add_argument("--variant", required=True)
    train.add_argument("--root", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--benchmark-train", type=Path, required=True)
    train.add_argument("--residual-checkpoint", type=Path, required=True)
    train.add_argument("--flow-checkpoint", type=Path, required=True)
    train.add_argument("--python-executable", required=True)
    train.add_argument("--tmp-rollout-dir", type=Path, required=True)
    train.add_argument("--fresh", action="store_true")
    train.add_argument("--format", choices=("shell", "nul"), default="shell")

    server = sub.add_parser("server-command")
    server.add_argument("--variant", required=True)
    server.add_argument("--root", type=Path, required=True)
    server.add_argument("--checkpoint", type=Path, required=True)
    server.add_argument("--format", choices=("shell", "nul"), default="shell")
    server.add_argument("--serve-prefix", nargs="+", required=True)

    agg = sub.add_parser("aggregate-phase-sr")
    agg.add_argument("--run-dir", type=Path, required=True)
    agg.add_argument("--shards", type=int, default=N_SHARDS)
    agg.add_argument("--rounds", type=int, default=PHASE_ROUNDS)
    agg.add_argument("--poses", type=int, default=BENCHMARK_POSE_CYCLE)
    agg.add_argument("--variant", default="flow_cfgrl")

    args = parser.parse_args(argv)
    if args.cmd == "aggregate-phase-sr":
        path = write_phase_sr_table(
            args.run_dir,
            n_shards=int(args.shards),
            n_rounds=int(args.rounds),
            n_poses=int(args.poses),
            variant=str(args.variant),
        )
        print(path.read_text(encoding="utf-8"), end="")
        return 0
    if args.cmd == "train-command":
        command = build_train_command(
            variant=args.variant,
            root=args.root,
            run_dir=args.run_dir,
            benchmark_train=args.benchmark_train,
            residual_checkpoint=args.residual_checkpoint,
            flow_checkpoint=args.flow_checkpoint,
            python_executable=args.python_executable,
            tmp_rollout_dir=args.tmp_rollout_dir,
            fresh=args.fresh,
        )
    else:
        command = build_server_command(
            variant=args.variant,
            root=args.root,
            checkpoint=args.checkpoint,
            serve_prefix=args.serve_prefix,
        )
    if args.format == "nul":
        sys.stdout.buffer.write(b"\0".join(c.encode() for c in command) + b"\0")
    else:
        print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
