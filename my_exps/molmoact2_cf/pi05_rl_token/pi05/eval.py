"""Evaluation, driven entirely by an `EvalConfig`.

    from pi05.config import EvalConfig
    from pi05.eval import evaluate
    evaluate(EvalConfig(scene="house21", chunk_size=1, episodes=36))

or from the shell:

    scripts/run_eval.py --scene house21

One call does all of it: bring up the frozen VLA on its own GPU, run the benchmark,
report the success rate with a confidence interval, and shut the server down again. The
last part is not optional -- a policy server left running holds a GPU and ~13 GB for
nothing, and the machine's owner kills idle ones on sight.

WHAT A NUMBER FROM HERE MEANS
-----------------------------
`success_rate` is the fraction of episodes where the task reported success. Episodes end
the moment they succeed (`end_on_success=True`), so the count is unambiguous. `ci95` is a
Wilson interval, which stays sane at the 0% and 100% ends where these scenes actually
live; a normal approximation would print negative lower bounds on house2.

Two runs are only comparable if they share `chunk_size` AND `conversion`. Both go into
the run's directory name and into `result.json` for exactly that reason. See
`pi05/README.md`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from .config import (
    CHECKPOINT,
    CODE,
    EvalConfig,
    HF_HOME,
    MLSPACES_ASSETS,
    MLSPACES_CACHE,
    OPENPI_CONFIG,
    RUN_CONFIG_ENV,
    TMP_ROLLOUT_DIR,
    VENV_TORCH,
    save_run_config,
)

log = logging.getLogger("pi05.eval")


# --------------------------------------------------------------------------------------
# Environment. Must run before anything imports molmo_spaces or torch.
# --------------------------------------------------------------------------------------


def prepare_environment(cfg: EvalConfig) -> Path:
    """Set what MolmoSpaces reads at import time, and publish the run config.

    Returns the path of the written config. Call this first, from the entry point,
    before any heavy import -- MuJoCo picks its EGL device and MolmoSpaces resolves its
    asset paths at import, so setting them afterwards is silently too late.
    """
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_run_config(cfg, run_dir / "config.json")

    os.environ[RUN_CONFIG_ENV] = str(config_path)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(cfg.egl_device)
    # Training runs the simulator and the learner in one process on `gpu`; evaluation can
    # put the simulator on a different card from the policy server, so it has its own.
    # Keep CUDA visible for collectors: NVIDIA EGL follows CUDA_VISIBLE_DEVICES, and
    # hiding the card made MuJoCo fall through to the macOS CGL renderer.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(cfg, "sim_gpu", cfg.gpu))
    os.environ["MLSPACES_ASSETS_DIR"] = str(MLSPACES_ASSETS)
    os.environ["MLSPACES_CACHE_DIR"] = str(MLSPACES_CACHE)
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    return config_path


# --------------------------------------------------------------------------------------
# The frozen VLA server
# --------------------------------------------------------------------------------------


def disable_policy_compile() -> None:
    """Run the frozen VLA eager.

    ``max-autotune`` CUDA-graph capture on the first /act fails with
    CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED when this GPU also has
    EGL collectors. ``TORCHINDUCTOR_CUDAGRAPHS=0`` is not enough: a cached
    inductor artifact still wraps the model in cudagraph_trees. Patching
    ``torch.compile`` to identity, before the policy is built, is the same
    switch as ``scripts/probe_divergence.py --no-compile``.
    """
    import torch

    torch.compile = lambda model, *args, **kwargs: model


def policy_server_env(cfg: EvalConfig) -> dict:
    """Environment for `serve_pi05_http.py`.

    `TORCHINDUCTOR_CUDAGRAPHS=0` is required when this process shares a GPU with EGL
    collectors: max-autotune CUDA-graph capture fails the first /act with
    CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED, and every later call dies
    on the leftover capture error.
    """
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": str(cfg.gpu),
        "HF_HOME": str(HF_HOME),
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": str(CODE),
        "TORCHINDUCTOR_CUDAGRAPHS": "0",
    }


def start_server(cfg: EvalConfig, log_path: Path) -> subprocess.Popen:
    """Launch the frozen pi0.5 on its own GPU.

    Its own process group, so shutting it down takes the whole tree with it rather than
    orphaning a child that keeps the card.
    """
    command = [
        str(VENV_TORCH),
        str(CODE / "scripts/serve_pi05_http.py"),
        "--checkpoint", str(cfg.checkpoint or CHECKPOINT),
        "--config", OPENPI_CONFIG,
        "--host", "127.0.0.1",
        "--port", str(cfg.port),
    ]
    if getattr(cfg, "record_tokens", ""):
        # The corpus is written by the server, in the same regime as the run that
        # produced it; the client just has to ask for tokens on every plan.
        command += [
            "--record-tokens", str(cfg.record_tokens),
            "--record-tag", cfg.scene or "val",
        ]
    env = policy_server_env(cfg)
    log.info("starting frozen pi0.5 on GPU %d, port %d", cfg.gpu, cfg.port)
    return subprocess.Popen(
        command,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(CODE),
        start_new_session=True,
    )


def stop_server(server: subprocess.Popen | None) -> None:
    """Always called, including on failure. An orphan holds a GPU until noticed."""
    if server is None or server.poll() is not None:
        return
    log.info("stopping the policy server")
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        server.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a rate. Sane at the 0% and 100% ends, unlike the normal one."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


def evaluate(cfg: EvalConfig) -> dict:
    """Run one evaluation and return its summary. Also written to result.json."""
    problem = cfg.validate()
    if problem:
        raise ValueError(f"this EvalConfig cannot be run: {problem}")

    run_dir = cfg.run_dir()
    if os.environ.get(RUN_CONFIG_ENV, "") != str(run_dir / "config.json"):
        raise RuntimeError(
            "prepare_environment(cfg) has not run for this config, so the simulator "
            "worker would read a different run config than the one being evaluated. "
            "Call it before importing molmo_spaces; scripts/run_eval.py does."
        )

    server = start_server(cfg, run_dir / "server.log")
    summary: dict = {
        "scene": cfg.scene or "val",
        "chunk_size": cfg.chunk_size,
        "conversion": cfg.conversion,
        "actor": cfg.actor,
        "token_ae": cfg.token_ae,
        "gate_step": cfg.gate_step if cfg.actor else None,
        "rl_action_space": cfg.rl_action_space if cfg.actor else None,
        "benchmark_dir": str(cfg.benchmark_dir()),
        "episodes_requested": cfg.episodes,
    }
    started = time.time()
    try:
        # Imported here, not at module scope: prepare_environment must have run first.
        from molmo_spaces.evaluation.eval_main import run_evaluation

        from .policy import Pi05EvalConfig

        results = run_evaluation(
            eval_config_cls=Pi05EvalConfig,
            benchmark_dir=cfg.benchmark_dir(),
            task_horizon_steps=cfg.horizon,
            num_workers=cfg.workers,
            use_wandb=False,
            max_episodes=cfg.episodes,
            output_dir=run_dir / "eval_output",
        )
        successes, total = int(results.success_count), int(results.total_count)
        low, high = wilson(successes, total)
        summary.update(
            successes=successes,
            episodes_run=total,
            success_rate=round(successes / total, 4) if total else None,
            ci95=[round(low, 4), round(high, 4)],
            output_dir=str(getattr(results, "output_dir", run_dir / "eval_output")),
        )
        if total != cfg.episodes:
            # Two different things land here and they need different reactions.
            #
            # MolmoSpaces may evaluate each episode spec several times -- the videos are
            # named batch_i_of_n -- so a total that is a whole multiple of what was asked
            # for is more rollouts, not fewer, and the rate is over all of them. Verified
            # by frame counts: two batches of one spec end at different steps, so they are
            # independent rollouts rather than halves of one.
            #
            # A total that is NOT a multiple means episodes went missing, which is the
            # failure that produced a wrong rate once already: two clients on one server
            # time out during the handshake and their episodes are skipped in silence.
            if cfg.episodes and total > cfg.episodes and total % cfg.episodes == 0:
                summary["batches_per_episode"] = total // cfg.episodes
                log.info(
                    "%d rollouts from %d episode specs (%d batches each)",
                    total, cfg.episodes, total // cfg.episodes,
                )
            else:
                summary["warning"] = (
                    f"{cfg.episodes} episodes requested but {total} rollouts ran, which is "
                    "not a whole multiple; episodes were skipped, usually because the "
                    "policy server timed out"
                )
                log.warning("%s", summary["warning"])
    finally:
        stop_server(server)
        summary["seconds"] = round(time.time() - started, 1)
        (run_dir / "result.json").write_text(json.dumps(summary, indent=2))

    if summary.get("episodes_run"):
        log.info(
            "SUCCESS RATE: %d/%d = %.1f%%  (95%% Wilson %.1f-%.1f%%)",
            summary["successes"],
            summary["episodes_run"],
            100 * summary["success_rate"],
            100 * summary["ci95"][0],
            100 * summary["ci95"][1],
        )
    else:
        log.error("no episodes completed; see %s", run_dir / "server.log")
    log.info("wrote %s", run_dir / "result.json")
    return summary


def cleanup_scratch() -> None:
    """Remove this project's episode scratch. Only ever touches TMP_ROLLOUT_DIR."""
    import shutil

    shutil.rmtree(TMP_ROLLOUT_DIR, ignore_errors=True)
