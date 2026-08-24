"""Eval configs for the fine-tuned pi0.5, kept in this project rather than MolmoSpaces.

eval_main takes a ``module:Class`` spec, so the one setting we need to change is made by
subclassing instead of editing molmo_spaces.

Why chunk_size=1. The model predicts joint deltas and the server turns them into the
absolute targets the simulator executes, using the arm state it was given. Only the state
at the start of a chunk is known, so element k of a longer chunk would be offset by a
stale pose. Measured on demonstrations, that error grows 0.027 rad per step and reaches
0.32 rad by k=15 -- larger than the motion being commanded. Re-inferring every step keeps
the conversion exact, at the cost of one forward pass per control step.

    PYTHONPATH=<rl_token> python molmo_spaces/evaluation/eval_main.py \
        pi05.eval_configs:Pi05PickEvalConfig --benchmark_dir ... --num_workers 1
"""

from __future__ import annotations

import os

from molmo_spaces.configs.policy_configs_baselines import PiPolicyConfig
from molmo_spaces.evaluation.configs.evaluation_configs import PiPolicyEvalConfig


class Pi05PickEvalConfig(PiPolicyEvalConfig):
    """pi0.5 served with delta-to-absolute conversion, so the chunk must be one step."""

    policy_config: PiPolicyConfig = PiPolicyConfig(
        # How many actions of the predicted chunk get executed before re-planning.
        # 1 matches how the model was trained -- each delta is labelled against its own
        # step's state, so applying delta[k] to the state from step t is wrong by exactly
        # the distance travelled since. Larger values are worth measuring anyway: the RL
        # harness ran at 8 and scored far higher, and that discrepancy is unexplained.
        chunk_size=int(os.environ.get("PI05_CHUNK_SIZE", "1")),
        # Demonstrations command the gripper fully open or fully closed and nothing in
        # between, so a threshold reproduces them exactly.
        grasping_type="binary",
        grasping_threshold=0.5,
        # 127.0.0.1 rather than "localhost": the name resolves to ::1 first and this
        # container has no IPv6, so connecting fails with EAFNOSUPPORT (errno 97).
        # Host and port come from the environment so the eval driver can pick a free
        # port instead of the config and the server disagreeing about which one to use.
        remote_config=dict(
            host=os.environ.get("PI05_POLICY_HOST", "127.0.0.1"),
            port=int(os.environ.get("PI05_POLICY_PORT", "8080")),
        ),
    )
