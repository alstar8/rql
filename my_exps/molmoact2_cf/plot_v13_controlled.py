"""Plot incomplete-safe V13 training/validation curves and JSON reports."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from v13_harness import (  # noqa: E402
    FINAL_VALIDATION_SEEDS,
    INTERIM_VALIDATION_SEEDS,
    MAX_VALID_EPISODES,
    SNAPSHOT_EPISODES,
    VALIDATION_EPISODES,
    VARIANTS,
    atomic_write_json,
    evaluation_complete,
)


_HERE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = _HERE / "runs" / "rlt_cf_v13_controlled"
DEFAULT_BENCHMARK_ROOT = _HERE / "runs" / "benchmarks" / "house0_kettle_v13"
COLORS = tuple(plt.get_cmap("tab10").colors)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        stream = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return rows
    with stream:
        for line in stream:
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                rows.append(candidate)
    return rows


def _finite_float(value: Any) -> float | None:
    try:
        rendered = float(value)
    except (TypeError, ValueError):
        return None
    return rendered if math.isfinite(rendered) else None


def discover_training_rows(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        variant_dir = run_dir / variant.name
        paths = sorted(variant_dir.rglob("metrics.jsonl")) if variant_dir.is_dir() else []
        rows = []
        for path in paths:
            for row in read_jsonl(path):
                copied = dict(row)
                copied["_source"] = str(path)
                rows.append(copied)
        rows.sort(
            key=lambda row: (
                int(row.get("valid_episodes", 0) or 0),
                int(row.get("env_steps", 0) or 0),
            )
        )
        output[variant.name] = rows
    return output


def _validation_identity(
    validation_root: Path,
    summary_path: Path,
) -> tuple[str, str, str, str] | None:
    try:
        relative = summary_path.relative_to(validation_root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 5:
        return None
    config, snapshot, policy, seed_name = parts[:4]
    if not snapshot.startswith("ep_") or not seed_name.startswith("seed_"):
        return None
    return config, snapshot, policy, seed_name


def discover_validation_runs(run_dir: Path) -> list[dict[str, Any]]:
    validation_root = run_dir / "validation"
    if not validation_root.is_dir():
        return []
    records = []
    summary_paths = sorted(validation_root.rglob("validation_summary.json"))
    for summary_path in summary_paths:
        identity = _validation_identity(validation_root, summary_path)
        if identity is None:
            continue
        config, snapshot, policy, seed_name = identity
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            continue
        out_dir = summary_path.parent
        try:
            episode = int(snapshot.removeprefix("ep_"))
            seed = int(seed_name.removeprefix("seed_"))
        except ValueError:
            continue
        records.append(
            {
                "config": config,
                "snapshot": snapshot,
                "episode": episode,
                "policy": policy,
                "seed": seed,
                "out_dir": str(out_dir),
                "complete": evaluation_complete(out_dir, VALIDATION_EPISODES),
                "summary": summary,
                "results": read_jsonl(out_dir / "validation_results.jsonl"),
            }
        )
    return records


def load_pose_metadata(benchmark_root: Path) -> dict[int, dict[str, Any]]:
    payload = read_json(benchmark_root / "val" / "benchmark.json")
    if not isinstance(payload, list):
        return {}
    output = {}
    for index, episode in enumerate(payload):
        if not isinstance(episode, dict):
            continue
        controlled = episode.get("controlled")
        task = episode.get("task")
        output[index] = {
            "episode_idx": index,
            "pair_id": controlled.get("pair_id") if isinstance(controlled, dict) else None,
            "kettle_pose_id": (
                controlled.get("kettle_pose_id")
                if isinstance(controlled, dict)
                else None
            ),
            "robot_pose_id": (
                controlled.get("robot_pose_id")
                if isinstance(controlled, dict)
                else None
            ),
            "pickup_obj_start_pose": (
                task.get("pickup_obj_start_pose")
                if isinstance(task, dict)
                else None
            ),
            "robot_base_pose": (
                task.get("robot_base_pose")
                if isinstance(task, dict)
                else None
            ),
        }
    return output


def build_per_pose_report(
    validation_runs: Iterable[dict[str, Any]],
    pose_metadata: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in validation_runs:
        for row in run["results"]:
            if not bool(row.get("valid", False)):
                continue
            try:
                episode_idx = int(row["episode_idx"])
            except (KeyError, TypeError, ValueError):
                continue
            grouped[
                (
                    run["config"],
                    run["snapshot"],
                    run["policy"],
                    episode_idx,
                )
            ].append(
                {
                    "seed": run["seed"],
                    "run_complete": bool(run["complete"]),
                    "success": bool(row.get("success", False)),
                    "episode_steps": int(
                        row.get("episode_steps", row.get("n_steps", 0)) or 0
                    ),
                }
            )

    records = []
    for key in sorted(grouped):
        config, snapshot, policy, episode_idx = key
        outcomes = grouped[key]
        successes = sum(int(outcome["success"]) for outcome in outcomes)
        records.append(
            {
                "config": config,
                "snapshot": snapshot,
                "policy": policy,
                "episode_idx": episode_idx,
                **pose_metadata.get(episode_idx, {}),
                "passes": len(outcomes),
                "successes": successes,
                "success_rate": successes / len(outcomes),
                "outcomes": outcomes,
            }
        )
    return {
        "schema_version": "v13-per-pose-1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": records,
    }


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _mcnemar_exact_p(actor_only: int, reference_only: int) -> float:
    discordant = int(actor_only) + int(reference_only)
    if discordant == 0:
        return 1.0
    lower = min(int(actor_only), int(reference_only))
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _paired_interval(differences: np.ndarray) -> tuple[float, float]:
    if differences.size == 0:
        return 0.0, 0.0
    mean = float(differences.mean())
    if differences.size == 1:
        return mean, mean
    standard_error = float(differences.std(ddof=1) / math.sqrt(differences.size))
    radius = 1.959963984540054 * standard_error
    return max(-1.0, mean - radius), min(1.0, mean + radius)


def _complete_outcomes(
    validation_runs: Iterable[dict[str, Any]],
    *,
    episode: int,
) -> dict[tuple[str, str, int, int], bool]:
    outcomes: dict[tuple[str, str, int, int], bool] = {}
    for run in validation_runs:
        if not run["complete"] or int(run["episode"]) != int(episode):
            continue
        for row in run["results"]:
            if not bool(row.get("valid", False)):
                continue
            try:
                episode_idx = int(row["episode_idx"])
            except (KeyError, TypeError, ValueError):
                continue
            outcomes[
                (
                    str(run["config"]),
                    str(run["policy"]),
                    int(run["seed"]),
                    episode_idx,
                )
            ] = bool(row.get("success", False))
    return outcomes


def _snapshot_gate_closed(run_dir: Path, config: str, episode: int) -> bool | None:
    metadata = read_json(
        run_dir / config / "snapshots" / f"ep_{episode:06d}" / "snapshot.json"
    )
    if not isinstance(metadata, dict):
        return None
    return not bool(
        metadata.get(
            "gate_deploy_actor",
            metadata.get("g_enabled", False),
        )
    )


def build_paired_policy_report(
    validation_runs: Iterable[dict[str, Any]],
    run_dir: Path,
    *,
    episode: int = 400,
) -> dict[str, Any]:
    runs = list(validation_runs)
    outcomes = _complete_outcomes(runs, episode=episode)
    comparisons: list[dict[str, Any]] = []
    invalid_cells: list[dict[str, Any]] = []
    warnings: list[str] = []

    for variant in VARIANTS:
        if variant.ae_trainable:
            unsafe = ["actor"]
            if variant.use_guide:
                unsafe.append("actor_guide")
            for policy in unsafe:
                invalid_cells.append(
                    {
                        "config": variant.name,
                        "snapshot_episode": int(episode),
                        "policy": policy,
                        "verdict": "invalid-by-construction",
                        "reason": (
                            "V13 AE weights were trained in raw robot coordinates, "
                            "but MolmoAct2 integrates in q01-q99-normalized coordinates."
                        ),
                    }
                )
            continue

        policy_pairs: list[tuple[str, str]] = []
        if "actor" in variant.policies:
            policy_pairs.append(("actor", "checkpoint_gate"))
        if "actor_guide" in variant.policies:
            policy_pairs.append(("actor_guide", "actor"))
        if "reference_noise" in variant.policies:
            policy_pairs.append(("reference_noise", "checkpoint_gate"))
        for treatment, control in policy_pairs:
            treatment_rows = {
                (seed, pose): success
                for (config, policy, seed, pose), success in outcomes.items()
                if config == variant.name and policy == treatment
            }
            control_rows = {
                (seed, pose): success
                for (config, policy, seed, pose), success in outcomes.items()
                if config == variant.name and policy == control
            }
            paired_keys = sorted(set(treatment_rows) & set(control_rows))
            if not paired_keys:
                continue
            treatment_values = np.asarray(
                [float(treatment_rows[key]) for key in paired_keys],
                dtype=np.float64,
            )
            control_values = np.asarray(
                [float(control_rows[key]) for key in paired_keys],
                dtype=np.float64,
            )
            differences = treatment_values - control_values
            treatment_successes = int(treatment_values.sum())
            control_successes = int(control_values.sum())
            treatment_interval = _wilson_interval(
                treatment_successes,
                len(paired_keys),
            )
            control_interval = _wilson_interval(control_successes, len(paired_keys))
            delta_interval = _paired_interval(differences)
            treatment_only = int(
                np.logical_and(treatment_values > 0.5, control_values <= 0.5).sum()
            )
            control_only = int(
                np.logical_and(treatment_values <= 0.5, control_values > 0.5).sum()
            )
            per_pose = []
            for pose in sorted({key[1] for key in paired_keys}):
                pose_keys = [key for key in paired_keys if key[1] == pose]
                pose_delta = float(
                    np.mean(
                        [
                            float(treatment_rows[key]) - float(control_rows[key])
                            for key in pose_keys
                        ]
                    )
                )
                per_pose.append(
                    {
                        "episode_idx": int(pose),
                        "passes": len(pose_keys),
                        "success_rate_delta": pose_delta,
                    }
                )
            comparisons.append(
                {
                    "config": variant.name,
                    "snapshot_episode": int(episode),
                    "treatment_policy": treatment,
                    "control_policy": control,
                    "paired_rollouts": len(paired_keys),
                    "treatment_successes": treatment_successes,
                    "treatment_success_rate": treatment_successes / len(paired_keys),
                    "treatment_wilson_95": list(treatment_interval),
                    "control_successes": control_successes,
                    "control_success_rate": control_successes / len(paired_keys),
                    "control_wilson_95": list(control_interval),
                    "paired_success_rate_delta": float(differences.mean()),
                    "paired_delta_95": list(delta_interval),
                    "treatment_only_successes": treatment_only,
                    "control_only_successes": control_only,
                    "mcnemar_exact_p": _mcnemar_exact_p(
                        treatment_only,
                        control_only,
                    ),
                    "per_pose": per_pose,
                }
            )

        if "checkpoint_gate" in variant.policies:
            gate_closed = _snapshot_gate_closed(run_dir, variant.name, episode)
            if gate_closed:
                warnings.append(
                    f"{variant.name}/checkpoint_gate is frozen-reference evaluation "
                    f"because the ep{episode} snapshot gate is closed."
                )

    complete_ep400_runs = sum(
        bool(run["complete"]) and int(run["episode"]) == int(episode)
        for run in runs
    )
    return {
        "schema_version": "v13-paired-policy-2",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_episode": int(episode),
        "complete_ep400_jobs": complete_ep400_runs,
        "expected_safe_ep400_jobs": 64,
        "expected_safe_ep400_rollouts": 64 * VALIDATION_EPISODES,
        "comparisons": comparisons,
        "invalid_by_construction": invalid_cells,
        "warnings": sorted(set(warnings)),
        "interpretation": (
            "actor and actor_guide are forced-policy counterfactuals; "
            "reference_noise is a matched exploration control; training "
            "collection success rate is not policy-efficacy evidence."
        ),
    }


def plot_paired_policy_deltas(
    paired_report: dict[str, Any],
    output_path: Path,
) -> None:
    comparisons = list(paired_report.get("comparisons", []))
    figure_height = max(4.0, 0.7 * len(comparisons) + 1.8)
    figure, axis = plt.subplots(figsize=(11, figure_height), dpi=150)
    if not comparisons:
        _placeholder(axis, "No complete paired policy comparisons")
    else:
        labels = [
            f"{row['config']}: {row['treatment_policy']} − {row['control_policy']}"
            for row in comparisons
        ]
        means = np.asarray(
            [row["paired_success_rate_delta"] for row in comparisons],
            dtype=np.float64,
        )
        intervals = np.asarray(
            [row["paired_delta_95"] for row in comparisons],
            dtype=np.float64,
        )
        errors = np.stack(
            [means - intervals[:, 0], intervals[:, 1] - means],
            axis=0,
        )
        y = np.arange(len(comparisons))
        axis.errorbar(
            means,
            y,
            xerr=errors,
            fmt="o",
            color="#1f77b4",
            capsize=4,
        )
        axis.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlim(-1.0, 1.0)
        axis.set_xlabel("Paired held-out success-rate delta (95% normal CI)")
        axis.grid(axis="x", alpha=0.25)
    axis.set_title("V13 ep400 forced-policy effects by identical pose/seed cell")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _placeholder(axis: plt.Axes, message: str) -> None:
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="#666666",
    )
    axis.set_xticks([])
    axis.set_yticks([])


def plot_training_sr(
    training_rows: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.5), dpi=150)
    plotted = False
    for index, variant in enumerate(VARIANTS):
        rows = training_rows.get(variant.name, [])
        x = []
        cumulative = []
        window = []
        for row in rows:
            episode = int(row.get("valid_episodes", 0) or 0)
            cumulative_value = _finite_float(row.get("cumulative_success_rate"))
            window_value = _finite_float(row.get("window_success_rate"))
            if cumulative_value is None:
                continue
            x.append(episode)
            cumulative.append(cumulative_value)
            window.append(window_value if window_value is not None else math.nan)
        if not x:
            continue
        color = COLORS[index % len(COLORS)]
        axis.plot(x, cumulative, color=color, linewidth=1.8, label=variant.name)
        axis.plot(x, window, color=color, linewidth=1.0, alpha=0.5, linestyle="--")
        plotted = True
    if not plotted:
        _placeholder(axis, "No complete training metric rows yet")
    else:
        axis.set_xlim(left=0, right=MAX_VALID_EPISODES)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Valid controlled-train episodes")
        axis.set_ylabel("Success rate")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2, loc="upper left")
    axis.set_title("V13 controlled training SR (solid cumulative, dashed 50-episode window)")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _validation_aggregates(
    validation_runs: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[int, list[float]]]:
    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in validation_runs:
        if not run["complete"]:
            continue
        success_rate = _finite_float(
            run["summary"].get("cumulative_success_rate")
        )
        if success_rate is None:
            continue
        grouped[(run["config"], run["policy"])][run["episode"]].append(success_rate)
    return grouped


def plot_validation_sr(
    validation_runs: list[dict[str, Any]],
    output_path: Path,
) -> None:
    aggregates = _validation_aggregates(validation_runs)
    figure, axes = plt.subplots(4, 2, figsize=(13, 15), dpi=150, sharex=True, sharey=True)
    for index, (variant, axis) in enumerate(zip(VARIANTS, axes.flat)):
        plotted = False
        for policy_index, policy in enumerate(variant.policies):
            values_by_episode = aggregates.get((variant.name, policy), {})
            xs = []
            means = []
            errors = []
            for episode in SNAPSHOT_EPISODES:
                values = values_by_episode.get(episode, [])
                if not values:
                    continue
                xs.append(episode)
                means.append(float(np.mean(values)))
                errors.append(
                    float(np.std(values, ddof=0) / math.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                )
            if not xs:
                continue
            axis.errorbar(
                xs,
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                linewidth=1.6,
                color=COLORS[policy_index % len(COLORS)],
                label=policy,
            )
            plotted = True
        if not plotted:
            _placeholder(axis, "No complete validation runs")
        else:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            axis.set_xticks(SNAPSHOT_EPISODES)
            axis.set_ylim(-0.03, 1.03)
        axis.set_title(variant.name, fontsize=10)
        if index % 2 == 0:
            axis.set_ylabel("Held-out success rate")
        if index >= 6:
            axis.set_xlabel("Immutable snapshot episode")
    figure.suptitle(
        "V13 held-out validation SR by forced policy\n"
        "(ep400 error bars are standard errors over four fixed seeds)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path)
    plt.close(figure)


def _plot_metric_series(
    axis: plt.Axes,
    training_rows: dict[str, list[dict[str, Any]]],
    keys: Sequence[str],
    title: str,
) -> bool:
    plotted = False
    for variant_index, variant in enumerate(VARIANTS):
        rows = training_rows.get(variant.name, [])
        for key_index, key in enumerate(keys):
            points = []
            for row in rows:
                value = _finite_float(row.get(key))
                if value is not None:
                    points.append((int(row.get("valid_episodes", 0) or 0), value))
            if not points:
                continue
            x, y = zip(*points)
            linestyle = ("-", "--", ":", "-.")[key_index % 4]
            label = variant.name if len(keys) == 1 else f"{variant.name}:{key}"
            axis.plot(
                x,
                y,
                color=COLORS[variant_index % len(COLORS)],
                linestyle=linestyle,
                linewidth=1.2,
                alpha=0.9,
                label=label,
            )
            plotted = True
    axis.set_title(title)
    if plotted:
        axis.grid(alpha=0.25)
        axis.set_xlim(left=0, right=MAX_VALID_EPISODES)
    else:
        _placeholder(axis, "No metric rows yet")
    return plotted


def plot_health(
    training_rows: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), dpi=150)
    _plot_metric_series(axes[0, 0], training_rows, ("q_td_loss",), "Critic TD loss")
    _plot_metric_series(
        axes[0, 1],
        training_rows,
        ("q_mean", "q_std"),
        "Critic endpoint mean / ensemble spread",
    )
    _plot_metric_series(axes[1, 0], training_rows, ("actor_adv",), "Actor advantage")
    _plot_metric_series(axes[1, 1], training_rows, ("ae_grad_norm",), "AE LoRA grad norm")
    _plot_metric_series(
        axes[2, 0],
        training_rows,
        ("guide_adv", "guide_w_norm", "guide_target_norm"),
        "Guide advantage / raw-target health",
    )
    _plot_metric_series(
        axes[2, 1],
        training_rows,
        ("gate_paired_lcb", "gate_sensitivity", "gate_critic_health"),
        "Gate paired LCB / sensitivity / critic health",
    )
    for axis in axes.flat:
        axis.set_xlabel("Valid controlled-train episodes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, fontsize=7, loc="lower center", ncol=4)
        figure.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def build_status_summary(
    run_dir: Path,
    training_rows: dict[str, list[dict[str, Any]]],
    validation_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    validation_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in validation_runs:
        validation_by_config[run["config"]].append(run)
    configs = {}
    for variant in VARIANTS:
        rows = training_rows.get(variant.name, [])
        latest = rows[-1] if rows else None
        snapshots = {}
        for episode in SNAPSHOT_EPISODES:
            source = run_dir / variant.name / "snapshots" / f"ep_{episode:06d}"
            required = [source / "rlt_cf.pt", source / "snapshot.json"]
            if variant.ae_trainable:
                required.append(source / "molmo_ae_lora.pt")
            snapshots[f"ep_{episode:06d}"] = all(path.is_file() for path in required)
        expected_runs = len(variant.policies) * (
            len(SNAPSHOT_EPISODES[:-1]) * len(INTERIM_VALIDATION_SEEDS)
            + len(FINAL_VALIDATION_SEEDS)
        )
        config_validation = validation_by_config.get(variant.name, [])
        complete_runs = sum(bool(run["complete"]) for run in config_validation)
        configs[variant.name] = {
            "latest_training_metrics": latest,
            "training_metrics_rows": len(rows),
            "training_complete": bool(
                latest
                and int(latest.get("valid_episodes", 0) or 0) >= MAX_VALID_EPISODES
            ),
            "snapshots": snapshots,
            "validation_expected_runs": expected_runs,
            "validation_discovered_runs": len(config_validation),
            "validation_complete_runs": complete_runs,
            "validation_complete": complete_runs == expected_runs,
        }
    return {
        "schema_version": "v13-status-1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "configs": configs,
        "training_complete_configs": sum(
            bool(config["training_complete"]) for config in configs.values()
        ),
        "validation_complete_configs": sum(
            bool(config["validation_complete"]) for config in configs.values()
        ),
    }


def generate_report(
    run_dir: Path,
    benchmark_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    run_dir = run_dir.resolve()
    benchmark_root = benchmark_root.resolve()
    plots_dir = (output_dir or (run_dir / "plots")).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    training_rows = discover_training_rows(run_dir)
    validation_runs = discover_validation_runs(run_dir)
    pose_metadata = load_pose_metadata(benchmark_root)

    paths = {
        "train_sr": plots_dir / "v13_train_sr.png",
        "validation_sr": plots_dir / "v13_validation_sr.png",
        "paired_deltas": plots_dir / "v13_paired_policy_deltas.png",
        "health": plots_dir / "v13_actor_critic_guide_health.png",
        "per_pose": plots_dir / "v13_per_pose_outcomes.json",
        "paired_policy": plots_dir / "v13_paired_policy_report.json",
        "status": plots_dir / "v13_status_summary.json",
    }
    paired_report = build_paired_policy_report(validation_runs, run_dir)
    plot_training_sr(training_rows, paths["train_sr"])
    plot_validation_sr(validation_runs, paths["validation_sr"])
    plot_paired_policy_deltas(paired_report, paths["paired_deltas"])
    plot_health(training_rows, paths["health"])
    atomic_write_json(
        paths["per_pose"],
        build_per_pose_report(validation_runs, pose_metadata),
    )
    atomic_write_json(paths["paired_policy"], paired_report)
    atomic_write_json(
        paths["status"],
        build_status_summary(run_dir, training_rows, validation_runs),
    )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = generate_report(args.run_dir, args.benchmark_root, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
