"""MolmoSpaces experiment config that runs our policy instead of the stock one.

Only the policy class differs from `MolmoAct2PolicyEvalConfig`: same Franka,
same cameras, same 66 ms control period, same `end_on_success`. A distinct class
also keeps eval output under its own directory name.

`run_evaluation` instantiates the config class with no arguments, and with more
than one worker each worker builds its own policy in its own process. So what
the policy needs beyond the class -- token AE path, actor checkpoint -- arrives
through the environment, which survives both fork and spawn.
"""

from __future__ import annotations

import os
from pathlib import Path

from molmo_spaces.configs.policy_configs_baselines import MolmoAct2PolicyConfig
from molmo_spaces.evaluation.configs.evaluation_configs import MolmoAct2PolicyEvalConfig
from molmo_spaces.policy.base_policy import PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient
from pydantic import Field

from .config import CHUNK
from .rl_policy import RLTokenPolicy

BENCHMARK_RELPATH = (
    "benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/"
    "FrankaPickDroidMiniBench_json_benchmark_20251231"
)


def publish_policy_env(token_ae: str, actor: str, host: str, port: int, device: str) -> None:
    """Hand the policy settings to worker processes started by run_evaluation."""
    os.environ["RLT_TOKEN_AE"] = token_ae
    os.environ["RLT_ACTOR"] = actor
    os.environ["RLT_SERVER_HOST"] = host
    os.environ["RLT_SERVER_PORT"] = str(port)
    os.environ["RLT_DEVICE"] = device


def default_benchmark_dir() -> Path:
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    path = ASSETS_DIR / BENCHMARK_RELPATH
    if not path.is_dir():
        raise FileNotFoundError(f"benchmark not found at {path}; pass --benchmark_dir")
    return path


class RLTokenPolicyConfig(MolmoAct2PolicyConfig):
    """Same wire protocol as the stock MolmoAct2 client, different policy class."""

    policy_cls: type = RLTokenPolicy
    policy_factory: PolicyFactory | None = None
    chunk_size: int = CHUNK
    stride: int = CHUNK  # evaluation stores nothing, so one decision per chunk
    token_ae: str = Field(default_factory=lambda: os.environ.get("RLT_TOKEN_AE", ""))
    actor: str = Field(default_factory=lambda: os.environ.get("RLT_ACTOR", ""))
    device: str = Field(default_factory=lambda: os.environ.get("RLT_DEVICE", "cuda:0"))
    remote_config: dict | None = Field(
        default_factory=lambda: {
            "host": os.environ.get("RLT_SERVER_HOST", "localhost"),
            "port": int(os.environ.get("RLT_SERVER_PORT", "8000")),
        }
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_factory is None:
            self.policy_factory = make_lenient(RLTokenPolicy)


class RLTokenEvalConfig(MolmoAct2PolicyEvalConfig):
    policy_config: RLTokenPolicyConfig = Field(default_factory=RLTokenPolicyConfig)
