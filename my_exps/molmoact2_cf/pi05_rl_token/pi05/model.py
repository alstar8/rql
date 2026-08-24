"""The policy: pi0.5 in, MolmoSpaces actions out, with RL Token optionally in between.

This is the one file that decides what the robot actually does. Every step of the action
path is here, so there is a single place to read when the arm misbehaves.

    scene images + instruction + joint state
              |
              v
    [ pi0.5 ]  ->  chunk of VLA_CHUNK actions, 8 dims each, in DELTA space
              |         and (for RL) the prefix token sequence, (968, 2048)
              v
    [ RL Token corrector ]  -> optionally replaces the chunk           (rl_token.py)
              |
              v
    [ ChunkExecutor ]  -> hands out chunk_size actions, one per env step,
              |            converting each delta to an ABSOLUTE joint target
              v
    MolmoSpaces executes {"arm": (7,) absolute radians, "gripper": 0 or 255}


WHAT pi0.5 IS
-------------
A two-expert flow-matching model:

    PaliGemma (gemma_2b, width 2048)   reads the two camera images and the instruction
    action expert (gemma_300m)         denoises the action chunk against that reading

One inference runs in two stages:

    1. prefix pass   embed_prefix(images, language) -> PaliGemma -> a KV cache.
                     Its last hidden state is a (968, 2048) token sequence: the model's
                     reading of the scene. RL Token uses exactly this as its RL state,
                     captured in vla_server.py. Measured deterministic across identical
                     calls, and sensitive to the image.
    2. denoising     10 flow-matching steps in the action expert against that cache,
                     producing VLA_CHUNK actions.

openpi pads actions to 32 dims and returns the first 8:

    [0:7]  joint deltas, one per Franka arm joint
    [7]    gripper, 0 = open, 1 = closed


WHY DELTAS, AND WHY THAT IS THE WHOLE BALLGAME
----------------------------------------------
The fine-tuning data labels every frame with ``q_commanded[t] - q_state[t]``, so an
action is a displacement from the state of its own step. The pretrained pi05_droid
checkpoint already spoke that language: probed with a real Pick state it returned +0.27
for a joint sitting at -2.09 rad, and its action norm statistics are zero-centred with
std ~0.1.

MolmoSpaces executes ABSOLUTE joint targets (``command_mode["arm"] = "joint_position"``),
so the conversion has to happen somewhere. It happens here and nowhere else.

Feeding raw deltas to the simulator commands the arm toward a folded pose on every step.
That is what the stock evaluation bridge does, and it is why the published pi0.5 baseline
on this benchmark reads 0/10. With the conversion in place the *unmodified* pretrained
checkpoint scores 2/8 on the same benchmark. The 0% was a bug, not a model.


CHUNK EXECUTION: chunk_size AND conversion
------------------------------------------
The model predicts VLA_CHUNK actions; the caller chooses how many to execute before
asking again. That choice matters more than anything else measured so far:

    chunk_size   house10   house21
             1     36.1%     88.9%
             2     50.0%     86.1%
             4     72.2%     77.8%
             8     83.3%     58.3%

36 episodes per cell, same checkpoint, nothing else changed. Both trends are significant
(p=0.0001, p=0.0066) and they point in opposite directions, so there is no single best
value. See config.DEFAULT_CHUNK_SIZE.

A long chunk is NOT wrong in itself. The label makes that precise. Each frame stores

    action[k] = q_commanded[t+k] - q_state[t+k]

a displacement measured at its own step -- semantically a joint velocity, which is why
openpi's DROID data config deliberately applies no further delta transform ("We assume
joint *velocity* actions, so we should *not* apply an additional delta transform",
openpi/src/openpi/training/config.py). A velocity chunk is meant to run open loop:
element k says "move at this rate at step t+k" and needs no particular absolute pose to
be meaningful. That is the ordinary pi0 / pi0.5 regime.

What that leaves is a second choice, and it is the one that actually decides correctness:
*which* arm state each delta is added to, since this environment executes absolute joint
targets rather than velocities.

    step_time    each action is added to the state observed on the step it executes,
                     command[t+k] = q_state[t+k] + action[k]
                 which reconstructs q_commanded[t+k] exactly, for every k. This is the
                 faithful reading of the label, at any chunk length.
    plan_time    all chunk_size actions are added to the state observed when the chunk
                 was planned,
                     command[t+k] = q_state[t] + action[k]
                                  = q_commanded[t+k] - (q_state[t+k] - q_state[t])
                 i.e. short by the distance travelled since the plan: 0.027 rad per step,
                 measured on demonstrations, reaching 0.32 rad by k=15.

So the drift usually blamed on "long chunks" belongs to plan_time specifically. plan_time
is exact only at k=0, which is why chunk_size=1 was called the safe setting -- under the
pre-refactor serving path plan_time was the only thing the code could do, so "chunk 1"
and "exact" happened to coincide.

The catch: EVERY measured number in this project, including the table above, used
plan_time, because the server converted once per plan and molmo_spaces' PI_Policy then
spends the chunk without re-converting. step_time has never been benchmarked. The
combination that should win on paper -- a long chunk with step_time -- is unmeasured, and
the fact that plan_time's lag *helps* house10 (36% -> 83%) is a warning against assuming
the faithful regime automatically wins.

At chunk_size == 1 the two are identical -- planning and execution happen at the same
step, against the same state. They diverge as the chunk grows. `test_pi05_model.py`
pins both, including their equivalence at chunk_size 1.
"""

from __future__ import annotations

import numpy as np

from .action_space import (
    ACTION_DIM,
    ARM_DOF,
    GRIPPER_CMD_MAX,
    GRIPPER_QPOS_OPEN,
    model_output_to_env_action,
    normalize_instruction,
)

#: Actions the model emits per call (openpi pi05_droid_finetune action_horizon).
ACTION_HORIZON = 16

__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ARM_DOF",
    "ChunkExecutor",
    "GRIPPER_CMD_MAX",
    "GRIPPER_QPOS_OPEN",
    "Pi05Policy",
    "observation_from_env",
    "to_env_action",
]


# --------------------------------------------------------------------------------------
# Building the model's input
# --------------------------------------------------------------------------------------


def observation_from_env(exterior_rgb, wrist_rgb, qpos: dict, instruction: str) -> dict:
    """Build the model input from what MolmoSpaces hands back.

    State is 8 numbers: the 7 arm joints, then the gripper opening scaled to 0..1 by
    GRIPPER_QPOS_OPEN. That constant is taken from
    molmo_spaces/policy/learned_policy/pi_policy.py so the state matches at eval time.

    Images arrive at 624x352 and are resized to 224x224 with padding inside openpi, the
    same geometry the training data was built with.
    """
    arm = np.asarray(qpos["arm"][:ARM_DOF], dtype=np.float32)
    gripper = np.clip(float(qpos["gripper"][0]) / GRIPPER_QPOS_OPEN, 0.0, 1.0)
    return {
        "observation/exterior_image_1_left": np.asarray(exterior_rgb, dtype=np.uint8),
        "observation/wrist_image_left": np.asarray(wrist_rgb, dtype=np.uint8),
        "observation/joint_position": arm,
        "observation/gripper_position": np.asarray([gripper], dtype=np.float32),
        # Every benchmark episode is phrased "pick up the <noun>." and the eval client
        # lowercases before sending, so the prompt is normalised to match.
        "prompt": normalize_instruction(instruction),
    }


# --------------------------------------------------------------------------------------
# Turning one predicted action into an environment command
# --------------------------------------------------------------------------------------


def to_absolute(action: np.ndarray, arm_state: np.ndarray) -> np.ndarray:
    """One predicted action with its arm deltas resolved against ``arm_state``.

    Returns the same 8-vector shape: absolute joint targets in [0:7], the gripper value
    untouched in [7] because the environment scaling happens later, once.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    arm_state = np.asarray(arm_state, dtype=np.float32).reshape(-1)[:ARM_DOF]
    if action.shape[0] < ACTION_DIM:
        raise ValueError(f"expected an action of >= {ACTION_DIM} dims, got {action.shape}")
    if arm_state.shape[0] != ARM_DOF:
        raise ValueError(f"expected {ARM_DOF} arm joints, got {arm_state.shape}")
    out = action[:ACTION_DIM].copy()
    out[:ARM_DOF] += arm_state
    return out


def to_env_action(
    action: np.ndarray,
    arm_state: np.ndarray,
    *,
    grasping: str = "binary",
    grasp_threshold: float = 0.5,
) -> dict:
    """Turn one predicted action into the dict MolmoSpaces executes.

    ``action`` is one row of a chunk: 7 joint deltas plus a gripper value in 0..1.
    ``arm_state`` is the arm position the delta is resolved against.
    """
    absolute = to_absolute(action, arm_state)
    return model_output_to_env_action(
        absolute,
        np.zeros(ARM_DOF, dtype=np.float32),  # already absolute; add nothing more
        grasping_type=grasping,
        grasping_threshold=grasp_threshold,
    )


# --------------------------------------------------------------------------------------
# Spending a chunk
# --------------------------------------------------------------------------------------


class ChunkExecutor:
    """Holds a predicted chunk and doles it out one action at a time.

    The caller asks the model for a chunk, then executes ``chunk_size`` of its actions
    before asking again. ``conversion`` decides which arm state each delta is resolved
    against; see the module docstring.

        executor = ChunkExecutor(chunk_size=4)
        for step in range(horizon):
            if executor.needs_new_chunk:
                executor.load(vla.predict(obs)["actions"], arm_state=current_arm)
            action = executor.next_action(current_arm)   # 8 dims, arm absolute
    """

    def __init__(self, chunk_size: int = 1, conversion: str = "plan_time") -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if conversion not in ("plan_time", "step_time"):
            raise ValueError(
                f"conversion must be 'plan_time' or 'step_time', got {conversion!r}"
            )
        self.chunk_size = chunk_size
        self.conversion = conversion
        self._chunk: np.ndarray | None = None
        self._plan_arm: np.ndarray | None = None
        self._index = 0

    @property
    def needs_new_chunk(self) -> bool:
        return self._chunk is None or self._index >= self.chunk_size

    @property
    def index(self) -> int:
        """How many actions of the current chunk have been handed out."""
        return self._index

    def load(self, chunk: np.ndarray, arm_state: np.ndarray) -> None:
        """Take a freshly predicted chunk, and the arm state it was planned from."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[1] < ACTION_DIM:
            raise ValueError(
                f"expected a (horizon, >={ACTION_DIM}) chunk, got {chunk.shape}"
            )
        if self.chunk_size > len(chunk):
            raise ValueError(
                f"chunk_size {self.chunk_size} exceeds the {len(chunk)} actions the model "
                "returned; the rest do not exist"
            )
        self._chunk = chunk
        self._plan_arm = np.asarray(arm_state, dtype=np.float32).reshape(-1)[:ARM_DOF].copy()
        self._index = 0

    def next_action(self, arm_state: np.ndarray) -> np.ndarray:
        """The next action of the chunk, arm resolved to absolute joint targets.

        ``arm_state`` is the arm position observed right now. It is used under
        "step_time"; under "plan_time" the state captured by `load` is used instead and
        this argument is ignored -- deliberately still required, so a caller cannot
        forget to observe the arm and silently get the wrong regime.
        """
        if self._chunk is None or self._plan_arm is None:
            raise RuntimeError("no chunk loaded; call load() first")
        if self._index >= self.chunk_size:
            raise RuntimeError(
                f"the chunk is spent ({self.chunk_size} actions handed out); check "
                "needs_new_chunk before asking for another"
            )
        action = self._chunk[self._index]
        self._index += 1
        against = self._plan_arm if self.conversion == "plan_time" else arm_state
        return to_absolute(action, against)

    def reset(self) -> None:
        self._chunk = None
        self._plan_arm = None
        self._index = 0


# --------------------------------------------------------------------------------------
# The whole policy, standalone
# --------------------------------------------------------------------------------------


class Pi05Policy:
    """Frozen pi0.5 driving the arm, with an optional RL-Token correction.

    This is the reference implementation of the control loop, independent of MolmoSpaces
    -- useful for tests and for reading. The class that actually runs inside the
    simulator is `pi05.policy.Pi05EvalPolicy`, which subclasses the MolmoSpaces policy so
    the evaluator can build it; it spends its chunk through this same `ChunkExecutor`.

    Without a corrector this is the plain VLA. With one, the corrector sees the scene
    tokens and adjusts the chunk before it is executed -- which is where RL Token enters
    the control path, and the only place it does.
    """

    def __init__(
        self,
        vla,
        chunk_size: int = 1,
        conversion: str = "plan_time",
        corrector=None,
        grasping: str = "binary",
        grasp_threshold: float = 0.5,
    ) -> None:
        self.vla = vla
        self.executor = ChunkExecutor(chunk_size, conversion)
        self.corrector = corrector
        self.grasping = grasping
        self.grasp_threshold = grasp_threshold
        self.step_index = 0
        self.last_tokens: np.ndarray | None = None

    def reset(self) -> None:
        self.executor.reset()
        self.step_index = 0
        self.last_tokens = None
        if self.corrector is not None:
            self.corrector.reset()

    def act(self, exterior_rgb, wrist_rgb, qpos: dict, instruction: str) -> dict:
        """One control step: observe, maybe re-plan, execute one action."""
        current_arm = np.asarray(qpos["arm"], dtype=np.float32).reshape(-1)[:ARM_DOF]

        if self.executor.needs_new_chunk:
            observation = observation_from_env(exterior_rgb, wrist_rgb, qpos, instruction)
            out = self.vla.predict(observation)
            chunk = np.asarray(out["actions"], dtype=np.float32)
            self.last_tokens = out.get("token_features")

            if self.corrector is not None:
                chunk = self.corrector.correct(
                    chunk=chunk,
                    tokens=self.last_tokens,
                    mask=out.get("token_attention_mask"),
                    step=self.step_index,
                    qpos=qpos,
                )
            self.executor.load(chunk, current_arm)

        absolute = self.executor.next_action(current_arm)
        self.step_index += 1
        return model_output_to_env_action(
            absolute,
            np.zeros(ARM_DOF, dtype=np.float32),  # already absolute
            grasping_type=self.grasping,
            grasping_threshold=self.grasp_threshold,
        )
