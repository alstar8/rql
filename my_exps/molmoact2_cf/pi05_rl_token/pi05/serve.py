"""Serving a fine-tuned pi0.5 to MolmoSpaces without touching the stock eval client.

The model predicts joint deltas (see action_space), but MolmoSpaces executes absolute
joint targets and molmo_spaces' pi_policy hands whatever it receives straight to the
environment. Rather than patch that file, which lives outside this project, the
conversion happens here: the server adds the arm state it was given to the predicted
delta and returns absolute targets, which is what the client already expects.

This is also the bug behind the existing zero-shot baseline. Served unmodified, the
pretrained checkpoint returns deltas near zero and the stock client reads them as
absolute targets, commanding the arm toward a folded pose on every step.

Chunk size must be 1 at eval. Element k of a chunk is a delta from the state at step
t+k, but only the state at t is known when the chunk is produced, so replaying a chunk
open-loop drifts by the distance already travelled: 0.027 rad per step, reaching 0.32 rad
by k=15, which exceeds the motion being commanded.
"""

from __future__ import annotations

import numpy as np
from openpi_client import base_policy as _base_policy

from .action_space import ARM_DOF


class DeltaToAbsolutePolicy(_base_policy.BasePolicy):
    """Wraps an openpi policy so its delta predictions come out as absolute targets."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        state_key: str = "observation/joint_position",
    ):
        self._policy = policy
        self._state_key = state_key

    def infer(self, obs: dict) -> dict:
        if self._state_key not in obs:
            raise KeyError(
                f"observation has no {self._state_key!r}; cannot turn a delta into an "
                "absolute joint target"
            )
        current_arm = np.asarray(obs[self._state_key], dtype=np.float32).reshape(-1)[:ARM_DOF]

        result = dict(self._policy.infer(obs))
        actions = np.array(result["actions"], dtype=np.float32, copy=True)

        # Every chunk element is offset by the same state, the only one known at
        # inference time. Exact for element 0 and drifting after it, which is why the
        # eval config must ask for a chunk size of 1.
        actions[..., :ARM_DOF] += current_arm

        # The gripper channel stays in 0..1 as predicted: the MolmoSpaces client is what
        # rescales it to the environment's 0..255, and doing it here too would double it.
        result["actions"] = actions
        return result

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()
