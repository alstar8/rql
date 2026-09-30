"""Per-round 36-episode probes: base G-off, gOn, student G-off.

Collectors read `eval_cmd.json` and `actor_limit` after every episode. The learner
holds them at each 50-episode boundary, saves the student, keeps the 500 best teacher
episodes, runs the three arms, and copies the student into the frozen expert only when
student G-off is strictly better than frozen G-off.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .collectors import FileCounter, read_inbox
from .distill_recorder import keep_best_trajectories
from .expert_swap import (
    frozen_checkpoint,
    reload_frozen_servers,
    request_student_save,
    snapshot_checkpoint,
)

log = logging.getLogger(__name__)

CMD_FILE = "eval_cmd.json"
LIMIT_FILE = "actor_limit"
EVAL_COUNTER = "eval.counter"
HEARTBEAT_DIR = "heartbeats"
ROUNDS_FILE = "rounds.jsonl"

ARMS = ("base_off", "gon", "student_off")


def cmd_path(out_dir) -> Path:
    return Path(out_dir) / CMD_FILE


def write_cmd(out_dir, phase: str, **extra) -> None:
    dest = cmd_path(out_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"phase": phase, **extra}
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(dest)


def read_cmd(out_dir) -> dict:
    path = cmd_path(out_dir)
    if not path.exists():
        return {"phase": "run"}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"phase": "run"}
    if not isinstance(data, dict):
        return {"phase": "run"}
    data.setdefault("phase", "run")
    return data


def write_actor_limit(out_dir, limit: int) -> None:
    path = Path(out_dir) / LIMIT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(str(int(limit)))
    tmp.replace(path)


def read_actor_limit(out_dir, default: int) -> int:
    path = Path(out_dir) / LIMIT_FILE
    if not path.exists():
        return int(default)
    raw = path.read_text().strip()
    return int(raw or default)


def next_actor_limit(probe_n: int, actor_episodes: int, swap_every: int, episodes: int) -> int:
    """Exclusive episode index (probe + actor) collectors may claim up to."""
    if swap_every <= 0:
        return probe_n + episodes
    end = ((actor_episodes // swap_every) + 1) * swap_every
    return probe_n + min(end, episodes)


def write_heartbeat(out_dir, rank: int, state: str, **extra) -> None:
    directory = Path(out_dir) / HEARTBEAT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"r{rank}.json"
    payload = {"rank": rank, "state": state, "ts": time.time(), **extra}
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(dest)


def read_heartbeats(out_dir) -> dict[int, dict]:
    directory = Path(out_dir) / HEARTBEAT_DIR
    if not directory.exists():
        return {}
    out: dict[int, dict] = {}
    for path in directory.glob("r*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "rank" in data:
            out[int(data["rank"])] = data
    return out


def wait_collectors_hold(
    out_dir,
    n_collectors: int,
    inbox,
    *,
    timeout_sec: float = 1800.0,
    poll_sec: float = 0.4,
) -> list[dict]:
    """Block until every collector is idle at the actor limit. Drain late actor pickles."""
    write_cmd(out_dir, "hold")
    deadline = time.time() + timeout_sec
    leftovers: list[dict] = []
    while time.time() < deadline:
        leftovers.extend(read_inbox(inbox))
        beats = read_heartbeats(out_dir)
        if len(beats) >= n_collectors and all(
            beats.get(rank, {}).get("state") == "hold" for rank in range(n_collectors)
        ):
            leftovers.extend(read_inbox(inbox))
            return leftovers
        time.sleep(poll_sec)
    raise TimeoutError(
        f"collectors did not hold within {timeout_sec:.0f}s "
        f"(heartbeats={ {k: v.get('state') for k, v in read_heartbeats(out_dir).items()} })"
    )


def reset_eval_counter(out_dir, seq: int | None = None) -> None:
    FileCounter(Path(out_dir) / EVAL_COUNTER).reset(0, seq=seq)


def summarize_arm(payloads: list[dict], n_eval: int) -> dict:
    n = len(payloads)
    successes = sum(int(p.get("success") or 0) for p in payloads)
    per_task: dict[str, list[int]] = {}
    for payload in payloads:
        scene = str(payload.get("scene") or "")
        bucket = per_task.setdefault(scene, [0, 0])
        bucket[0] += 1
        bucket[1] += int(payload.get("success") or 0)
    return {
        "n": n,
        "n_requested": n_eval,
        "successes": successes,
        "sr": round(successes / max(n, 1), 4),
        "incomplete": n < n_eval,
        "per_task": {
            name: {"episodes": count, "successes": s, "sr": round(s / max(count, 1), 4)}
            for name, (count, s) in sorted(per_task.items())
        },
    }


def apply_arm(policy, arm: str) -> None:
    """Switch a live collector policy between training and a round-eval arm."""
    if arm == "run":
        policy.use_actor = True
        policy.record_distill = True
        if policy.corrector is not None:
            policy.corrector.explore = True
        return
    policy.record_distill = False
    if arm in ("base_off", "student_off"):
        policy.use_actor = False
        return
    if arm == "gon":
        policy.use_actor = True
        if policy.corrector is not None:
            policy.corrector.explore = False
        return
    raise ValueError(f"unknown round-eval arm {arm!r}")


def _wait_hold_states(out_dir, n_collectors: int, timeout_sec: float = 1800.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        beats = read_heartbeats(out_dir)
        if len(beats) >= n_collectors and all(
            beats.get(rank, {}).get("state") == "hold" for rank in range(n_collectors)
        ):
            return
        time.sleep(0.2)
    raise TimeoutError("collectors did not return to hold between round-eval arms")


def merge_arm_payloads(
    got: list[dict], incoming: list[dict], *, arm: str, seq: int
) -> list[dict]:
    """Keep the first payload per eval index; ignore other arms and duplicates."""
    seen: dict[int, dict] = {int(payload["episode"]): payload for payload in got}
    for payload in incoming:
        if (
            payload.get("round_eval")
            and payload.get("arm") == arm
            and int(payload.get("seq") or 0) == seq
        ):
            idx = int(payload.get("episode"))
            if idx in seen:
                log.warning("duplicate %s eval_i=%s", arm, idx)
                continue
            seen[idx] = payload
        else:
            log.warning(
                "dropping non-%s payload during round eval: keys=%s",
                arm,
                sorted(payload),
            )
    return list(seen.values())


def _run_arm(
    out_dir,
    *,
    arm: str,
    n_eval: int,
    seq: int,
    n_collectors: int,
    inbox,
    timeout_sec: float,
) -> list[dict]:
    write_cmd(out_dir, "hold")
    _wait_hold_states(out_dir, n_collectors, timeout_sec=timeout_sec)
    reset_eval_counter(out_dir, seq=seq)
    write_cmd(out_dir, "round_eval", arm=arm, n=n_eval, seq=seq)
    got: list[dict] = []
    deadline = time.time() + timeout_sec
    while len(got) < n_eval:
        if time.time() > deadline:
            missing = sorted(set(range(n_eval)) - {int(p["episode"]) for p in got})
            beats = {
                k: {"state": v.get("state"), "eval_i": v.get("eval_i")}
                for k, v in read_heartbeats(out_dir).items()
            }
            log.warning(
                "%s probe timed out at %d/%d missing=%s heartbeats=%s; using partial",
                arm, len(got), n_eval, missing, beats,
            )
            break
        got = merge_arm_payloads(got, read_inbox(inbox), arm=arm, seq=seq)
        time.sleep(0.2)
    write_cmd(out_dir, "hold")
    try:
        _wait_hold_states(out_dir, n_collectors, timeout_sec=min(timeout_sec, 1800.0))
    except TimeoutError:
        log.warning("collectors did not return to hold after %s; continuing", arm)
    return got[:n_eval]


def wipe_distill_shards(distill_out) -> int:
    """Delete every teacher shard. Prefer keep_best_trajectories in the live loop."""
    return int(keep_best_trajectories(distill_out, keep=0)["deleted"])


def student_beats_frozen(student_off: dict, base_off: dict) -> bool:
    """Copy the student into ã only when the G-off probe is strictly better."""
    if student_off.get("incomplete") or base_off.get("incomplete"):
        return False
    if int(student_off.get("n") or 0) <= 0 or int(base_off.get("n") or 0) <= 0:
        return False
    return float(student_off["sr"]) > float(base_off["sr"])


def append_round(out_dir, record: dict) -> None:
    path = Path(out_dir) / ROUNDS_FILE
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def run_round_probes_and_swap(
    *,
    out_dir,
    cfg,
    n_collectors: int,
    inbox,
    n_eval: int,
    actor_episodes: int,
    student,
    generation: int,
) -> tuple[int, dict]:
    """Save student, keep the best teacher episodes, probe, copy only if student > frozen."""
    leftovers = wait_collectors_hold(out_dir, n_collectors, inbox)
    if leftovers:
        log.warning("round eval: %d inbox items while holding collectors", len(leftovers))
    if student is None or student.poll() is not None:
        raise RuntimeError("student trainer is not running at round-eval time")
    round_dir = Path(cfg.distill_student_dir).parent / "student_round"
    frozen_dir = Path(cfg.distill_frozen_dir) if cfg.distill_frozen_dir else (
        Path(cfg.distill_student_dir).parent / "frozen_expert"
    )
    status = request_student_save(
        cfg.distill_control_dir,
        previous_generation=generation,
        is_alive=lambda: student.poll() is None,
        student_dir=cfg.distill_student_dir,
        snapshot_dir=round_dir,
    )
    keep = int(getattr(cfg, "distill_keep_best", 500) or 0)
    pruned = keep_best_trajectories(cfg.distill_out, keep=keep)
    seq0 = actor_episodes * 10
    timeout = max(1800.0, 90.0 * n_eval)
    base_off = _run_arm(
        out_dir, arm="base_off", n_eval=n_eval, seq=seq0, n_collectors=n_collectors,
        inbox=inbox, timeout_sec=timeout,
    )
    gon = _run_arm(
        out_dir, arm="gon", n_eval=n_eval, seq=seq0 + 1, n_collectors=n_collectors,
        inbox=inbox, timeout_sec=timeout,
    )
    ports = [port for _, port in cfg.server_endpoints()]
    reload_frozen_servers(ports, str(round_dir))
    student_off = _run_arm(
        out_dir, arm="student_off", n_eval=n_eval, seq=seq0 + 2, n_collectors=n_collectors,
        inbox=inbox, timeout_sec=timeout,
    )
    base_summary = summarize_arm(base_off, n_eval)
    student_summary = summarize_arm(student_off, n_eval)
    copied = student_beats_frozen(student_summary, base_summary)
    if copied:
        snapshot_checkpoint(round_dir, frozen_dir)
        log.info(
            "copy student -> frozen: student_off %.1f%% > base_off %.1f%%",
            100 * student_summary["sr"], 100 * base_summary["sr"],
        )
    else:
        restore = frozen_checkpoint(cfg)
        reload_frozen_servers(ports, restore)
        log.info(
            "keep frozen ã (%s): student_off %.1f%% vs base_off %.1f%%%s",
            restore,
            100 * student_summary["sr"],
            100 * base_summary["sr"],
            " (incomplete probe)" if student_summary.get("incomplete") or base_summary.get("incomplete") else "",
        )
    record = {
        "round": actor_episodes // max(int(cfg.distill_swap_every), 1),
        "ep_end": actor_episodes,
        "generation": int(status.get("generation") or 0),
        "student_step": status.get("step"),
        "student_loss": status.get("loss"),
        "base_off": base_summary,
        "gon": summarize_arm(gon, n_eval),
        "student_off": student_summary,
        "copied": copied,
        "keep_best": pruned,
    }
    append_round(out_dir, record)
    log.info(
        "round %s ep %d: base_off %.1f%%  gOn %.1f%%  student_off %.1f%%  copied=%s  (n=%d%s)",
        record["round"],
        actor_episodes,
        100 * record["base_off"]["sr"],
        100 * record["gon"]["sr"],
        100 * record["student_off"]["sr"],
        copied,
        n_eval,
        "" if not any(record[a].get("incomplete") for a in ARMS) else " incomplete",
    )
    return int(status["generation"]), record
