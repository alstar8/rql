"""Where RL Token enters the control path, and nothing else.

The frozen VLA proposes a chunk. RL Token may replace it with a refined one. That
replacement happens here, in `RLTokenCorrector.correct`, which is the single point of
contact between the RL machinery in `rlt/` and the action path in `pi05/model.py`.

    VLA chunk (deltas)  --+--> gate closed --> executed unchanged
                          |
                          +--> gate open ---> encode tokens -> z_rl
                                              build x = [z_rl, proprio]
                                              a = pi(x, a_ref)
                                              executed instead


THE GATE
--------
Default: RL drives from env step 0. Set `gate_step=N` to let the frozen VLA run the
first N steps (the old "approach prefix"), `gate_frac=0.1` for ~10% of the horizon
snapped down to a chunk boundary (48 of 500, 40 of 400), or `gate_step=-1` to use
the scene catalog handover (mug 56, kettle 40, …). Once open, a latched gate stays
open for the rest of the episode. `gate_frac > 0` wins over `gate_step`.

A distance gate was tried first and abandoned for a measured reason -- the evaluator
records the object's pose from the benchmark spec, not its live position, so the number
stops describing reality the moment the policy nudges the object, and on an elongated
object the centre stays far while the gripper is already at a graspable end. See
`pi05/proximity.py`, which keeps both gates and the measurements behind them.


WHICH SPACE THE ACTOR REFINES
-----------------------------
`rl_action_space` decides what the reference chunk means to the actor:

    "absolute"  the reference is absolute joint targets. The correction is converted back
                to deltas before it is handed on, because `ChunkExecutor` converts again;
                under plan_time conversion that round trip is exact.
    "delta"     the reference is the raw joint deltas the model emits, and the correction
                is passed straight through.

The distinction matters because the actor is a plain MLP that emits the chunk outright
(`rlt/networks.py`), not a residual on the reference. At initialisation it outputs values
near zero: near-zero deltas barely move the arm, near-zero absolute targets command a
completely different pose. sigma and beta are scale-sensitive for the same reason. The
discarded 8-run matrix ran in "absolute", which is why it is still the default.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .config import ACTION_DIM, ARM_DOF, PROPRIO_DIM, TORCH_DEVICE, VLA_TOKEN_DIM

log = logging.getLogger(__name__)

# rlt reads its token width from the environment at import time and defaults to
# MolmoAct2's 2560. Every pi0.5 run is 2048, so it is set here rather than left to a
# launcher to remember -- forgetting it makes rlt reject the corpus with a shape error
# far away from the cause.
_declared = os.environ.get("RLT_VLA_TOKEN_DIM")
if _declared not in (None, str(VLA_TOKEN_DIM)):
    log.warning("RLT_VLA_TOKEN_DIM was %s; pi0.5 is %d, overriding", _declared, VLA_TOKEN_DIM)
os.environ["RLT_VLA_TOKEN_DIM"] = str(VLA_TOKEN_DIM)


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------


class StepGate:
    """Decides, per env step, whether RL or the VLA drives.

    Latched (the default, and what the paper does): control is handed over once at
    `step`, not traded back and forth. `step=0` (the default) means RL drives from the
    first env step. Unlatched means the single step `step` only, which exists for
    symmetry with the distance gate and is unlikely to be what anyone wants.
    """

    def __init__(self, step: int = 0, latch: bool = True) -> None:
        if step < 0:
            raise ValueError(f"gate step must be >= 0, got {step}")
        self.step = int(step)
        self.latch = bool(latch)

    def is_open(self, t: int) -> bool:
        return t >= self.step if self.latch else t == self.step

    def mask(self, n_steps: int) -> np.ndarray:
        """The whole episode's gate as a boolean mask, for plots and tests."""
        steps = np.arange(int(n_steps))
        return steps >= self.step if self.latch else steps == self.step

    def __repr__(self) -> str:
        return f"StepGate(step={self.step}, latch={self.latch})"


# --------------------------------------------------------------------------------------
# The correction
# --------------------------------------------------------------------------------------


class RLTokenCorrector:
    """Replaces the VLA's chunk with the actor's, once the gate is open.

    Everything it needs beyond the chunk arrives per call: the prefix tokens that become
    the RL state, the proprioception that completes it, the step index for the gate, and
    the arm state for the action-space conversion.
    """

    def __init__(
        self,
        agent,
        encoder,
        chunk_size: int,
        gate: StepGate,
        *,
        action_space: str = "absolute",
        explore: bool = False,
        record_deployed: bool = False,
    ) -> None:
        if action_space not in ("absolute", "delta"):
            raise ValueError(f"action_space must be 'absolute' or 'delta', got {action_space!r}")
        self.agent = agent
        self.encoder = encoder
        self.chunk_size = int(chunk_size)
        self.gate = gate
        self.action_space = action_space
        self.explore = explore
        # V25 distillation wants the *deployed* chunk as its target, not the exploring
        # one that gets executed. Collectors must keep exploring for RL to learn, but
        # the policy whose success rate is being preserved is the greedy one evaluation
        # runs (`explore=False`, which on the flow agents means the EMA target net).
        # Distilling the executed chunk would fit the exploration noise instead.
        self.record_deployed = bool(record_deployed)
        self.corrections = 0
        # The last decision's RL state and reference chunk. Training records these; a
        # plain evaluation ignores them. They are set whenever tokens were available,
        # including on steps where the gate kept the VLA in control.
        self.last_state: np.ndarray | None = None
        self.last_reference: np.ndarray | None = None
        self.last_tokens: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.last_log_prob: float | None = None
        #: The deployed (greedy) chunk for the last correction, in delta space. Only
        #: populated when `record_deployed`; this is the V25 distillation target.
        self.last_deployed: np.ndarray | None = None

    def reset(self) -> None:
        self.corrections = 0
        self.last_state = None
        self.last_reference = None
        self.last_tokens = None
        self.last_mask = None
        self.last_log_prob = None
        self.last_deployed = None

    def describe(self) -> str:
        return (
            f"{self.gate}, refining {self.action_space} actions, "
            f"chunk_size={self.chunk_size}, explore={self.explore}"
        )

    # --- the RL state ---------------------------------------------------------------

    def rl_state(self, tokens: np.ndarray, mask: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        """x = [z_rl, proprio]: the compressed scene reading plus joint positions and
        velocities. This is the only thing the actor and critic ever see of the world."""
        proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if proprio.shape != (PROPRIO_DIM,):
            raise ValueError(f"proprio must be ({PROPRIO_DIM},), got {proprio.shape}")
        z_rl = self.encoder.encode(tokens, mask)
        return np.concatenate([z_rl, proprio]).astype(np.float32)

    # --- action space ---------------------------------------------------------------

    def reference_from_chunk(self, chunk: np.ndarray, arm_state: np.ndarray) -> np.ndarray:
        """The reference the actor is shown, flattened, in the configured space."""
        rows = np.asarray(chunk, dtype=np.float32)[: self.chunk_size, :ACTION_DIM].copy()
        if self.action_space == "absolute":
            rows[:, :ARM_DOF] += np.asarray(arm_state, dtype=np.float32)[:ARM_DOF]
        return rows.reshape(-1)

    def chunk_from_action(self, action: np.ndarray, arm_state: np.ndarray) -> np.ndarray:
        """Invert `reference_from_chunk`: back to the delta space the executor expects.

        The executor converts deltas to absolute targets. Under "absolute" the actor has
        already produced absolute targets, so the arm state is subtracted here and added
        back there -- exact under plan_time conversion, which is why the two settings are
        required to agree (see config.RLConfig.validate).
        """
        rows = np.asarray(action, dtype=np.float32).reshape(self.chunk_size, ACTION_DIM).copy()
        if self.action_space == "absolute":
            rows[:, :ARM_DOF] -= np.asarray(arm_state, dtype=np.float32)[:ARM_DOF]
        return rows

    # --- the hook the policy calls ---------------------------------------------------

    def correct(
        self,
        *,
        chunk: np.ndarray,
        tokens: np.ndarray | None,
        mask: np.ndarray | None,
        step: int,
        proprio: np.ndarray,
        arm_state: np.ndarray,
        use_actor: bool = True,
        **_ignored,
    ) -> np.ndarray:
        """The VLA's chunk in, the chunk to execute out. Deltas on both sides.

        `use_actor` is how the trainer holds the actor back during warmup: the decision
        is still recorded, but the VLA's own chunk is what runs.
        """
        self.last_state = None
        self.last_reference = None
        self.last_tokens = None
        self.last_mask = None
        self.last_log_prob = None
        self.last_deployed = None
        active = use_actor and self.gate.is_open(step)

        if tokens is None or mask is None:
            if active:
                raise RuntimeError(
                    "the gate is open but the server returned no tokens, so the RL state "
                    "cannot be built. The client must request them (want_tokens=True)."
                )
            return chunk

        # Recorded even when the VLA stays in control, so warmup episodes and the
        # pre-gate stretch can still fill the replay buffer.
        self.last_state = self.rl_state(tokens, mask, proprio)
        self.last_reference = self.reference_from_chunk(chunk, arm_state)
        self.last_tokens = np.asarray(tokens, dtype=np.float16)
        self.last_mask = np.asarray(mask, dtype=np.float16)
        if not active:
            return chunk

        action = self.agent.act(self.last_state, self.last_reference, explore=self.explore)
        self.last_log_prob = getattr(self.agent, "last_log_prob", None)
        self.corrections += 1
        executed = self.chunk_from_action(action, arm_state)
        if self.record_deployed:
            if self.explore:
                # A second, noise-free pass at the same state. Cheap next to the VLA
                # call that produced the reference: a 10-step MLP unroll.
                deployed = self.agent.act(self.last_state, self.last_reference, explore=False)
                self.last_deployed = self.chunk_from_action(deployed, arm_state)
            else:
                self.last_deployed = executed
        return executed


# --------------------------------------------------------------------------------------
# Building one from a config
# --------------------------------------------------------------------------------------


def corrector_from_config(run, *, explore: bool = False) -> RLTokenCorrector:
    """Load the phase-1 encoder and the trained actor named by an EvalConfig."""
    import torch

    from rlt.consensusflow import make_agent
    from rlt.config import OnlineConfig
    from rlt.token_ae import RLTokenAE
    from rlt.vla import TokenEncoder

    # cuda:0 regardless of run.gpu: CUDA_VISIBLE_DEVICES already isolated the card.
    device = TORCH_DEVICE if torch.cuda.is_available() else "cpu"
    encoder = TokenEncoder(str(run.token_ae), device)

    checkpoint = torch.load(str(run.actor), map_location="cpu", weights_only=False)
    stored = checkpoint.get("config", {})
    cfg = OnlineConfig(
        **{k: v for k, v in stored.items() if k in OnlineConfig.__dataclass_fields__}
    )
    cfg.device = device
    cfg.algorithm = checkpoint.get("algorithm", stored.get("algorithm", "rl_token"))

    chunk_size = int(run.chunk_size)
    token_ae = None
    if cfg.algorithm in ("consensusflow", "flow_rlt", "v22_24", "v22_25"):
        token_ae = RLTokenAE.load(str(run.token_ae), map_location=device)
    agent = make_agent(
        cfg,
        encoder.z_dim + PROPRIO_DIM,
        chunk_size * ACTION_DIM,
        chunk_size,
        token_ae=token_ae,
    )
    agent.load(str(run.actor))
    # Paired-intervention hook: an explicit eval-time lambda overrides the
    # checkpoint's trained guidance scale (0 = guide off).
    guidance_coef = float(getattr(run, "guidance_coef", -1.0))
    if guidance_coef >= 0 and hasattr(agent, "set_guidance_coef"):
        agent.set_guidance_coef(guidance_coef)

    return RLTokenCorrector(
        agent,
        encoder,
        chunk_size,
        # resolved, so the actor is evaluated at the gate it was trained at.
        StepGate(run.resolved_gate_step(), run.gate_latch),
        action_space=run.rl_action_space,
        explore=explore,
    )
