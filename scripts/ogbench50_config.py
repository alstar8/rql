"""OGBench 50-task benchmark configuration (RQL paper / Li & Levine 2024)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
TASKS_PATH = BENCHMARK_DIR / "ogbench50_tasks.json"
BASELINES_PATH = BENCHMARK_DIR / "paper_figure5_baselines.json"


@dataclass(frozen=True)
class OGBench50Task:
    task_id: str
    domain_id: str
    task_num: int
    env_name: str
    dataset_name: str
    sparse: bool
    hyperparams: dict[str, Any]
    dataset_100m_dir_env: str | None = None

    @property
    def run_group(self) -> str:
        return run_group_for_task(self, method="baseline")

    def run_group_for(self, method: str) -> str:
        return run_group_for_task(self, method=method)

    @property
    def log_stem(self) -> str:
        return self.task_id.replace("-", "_")


def load_tasks_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or TASKS_PATH) as f:
        return json.load(f)


def expand_tasks(config: dict[str, Any] | None = None) -> list[OGBench50Task]:
    if config is None:
        config = load_tasks_config()
    tasks: list[OGBench50Task] = []
    for domain in config["domains"]:
        dtype = domain["dataset_type"]
        for task_num in range(1, 6):
            env_name = f"{domain['id']}-{dtype}-singletask-task{task_num}-v0"
            task_id = f"{domain['id']}-task{task_num}"
            tasks.append(
                OGBench50Task(
                    task_id=task_id,
                    domain_id=domain["id"],
                    task_num=task_num,
                    env_name=env_name,
                    dataset_name=domain["dataset_name"],
                    sparse=bool(domain.get("sparse", False)),
                    hyperparams=dict(domain["hyperparams"]),
                    dataset_100m_dir_env=domain.get("dataset_100m_dir_env"),
                )
            )
    return tasks


def unique_dataset_names(config: dict[str, Any] | None = None) -> list[str]:
    if config is None:
        config = load_tasks_config()
    return sorted({domain["dataset_name"] for domain in config["domains"]})


def run_group_for_task(task: OGBench50Task, method: str = "baseline") -> str:
    stem = task.task_id.replace("-", "_")
    if method == "baseline":
        return f"ogbench50-{stem}"
    return f"ogbench50-{method}-{stem}"


def build_main_args(
    task: OGBench50Task,
    seed: int,
    save_dir: str,
    ogbench_data_dir: str,
    offline_steps: int = 1_000_000,
    batch_size: int = 256,
    agent: str = "agents/rql.py",
    method: str = "baseline",
    guidance_coef: float | None = None,
    distill_coef: float | None = None,
    consensus_tau: float | None = None,
    uncertainty_coef: float | None = None,
    consensus_floor: float | None = None,
    conflict_power: float | None = None,
    residual_coef: float | None = None,
) -> list[str]:
    hp = task.hyperparams
    args = [
        "main.py",
        f"--agent={agent}",
        f"--env_name={task.env_name}",
        f"--run_group={run_group_for_task(task, method)}",
        f"--save_dir={save_dir}",
        f"--seed={seed}",
        f"--ogbench_data_dir={ogbench_data_dir}",
        f"--offline_steps={offline_steps}",
        "--online_steps=0",
        f"--agent.batch_size={batch_size}",
        f"--agent.alpha={hp['alpha']}",
        f"--agent.expectile={hp['expectile']}",
        f"--agent.ensemble_ct={hp['ensemble_ct']}",
        f"--agent.rho={hp['rho']}",
        f"--agent.h={hp['h']}",
        f"--agent.discount={hp['discount']}",
    ]
    if task.sparse:
        args.append("--sparse")
    if task.dataset_100m_dir_env:
        dataset_dir = os.environ.get(task.dataset_100m_dir_env, "")
        if not dataset_dir:
            raise RuntimeError(
                f"Task {task.env_name} requires ${task.dataset_100m_dir_env} to be set."
            )
        args.append(f"--ogbench_dataset_dir={dataset_dir}")
    if guidance_coef is not None:
        args.append(f"--agent.guidance_coef={guidance_coef}")
    if distill_coef is not None:
        args.append(f"--agent.distill_coef={distill_coef}")
    if consensus_tau is not None:
        args.append(f"--agent.consensus_tau={consensus_tau}")
    if uncertainty_coef is not None:
        args.append(f"--agent.uncertainty_coef={uncertainty_coef}")
    if consensus_floor is not None:
        args.append(f"--agent.consensus_floor={consensus_floor}")
    if conflict_power is not None:
        args.append(f"--agent.conflict_power={conflict_power}")
    if residual_coef is not None:
        args.append(f"--agent.residual_coef={residual_coef}")
    return args


def is_run_complete(
    exp_root: Path,
    seed: int,
    final_step: int = 1_000_000,
) -> bool:
    import csv

    pattern = f"sd{seed:03d}_*"
    for run_dir in exp_root.glob(pattern):
        eval_csv = run_dir / "eval.csv"
        if not eval_csv.is_file():
            continue
        with eval_csv.open() as f:
            reader = csv.DictReader(f)
            steps = [int(float(row["step"])) for row in reader]
        if steps and max(steps) >= final_step:
            return True
    return False


def exp_root_for_task(
    save_dir: str,
    task: OGBench50Task,
    method: str = "baseline",
) -> Path:
    return Path(save_dir) / "rql" / run_group_for_task(task, method)
