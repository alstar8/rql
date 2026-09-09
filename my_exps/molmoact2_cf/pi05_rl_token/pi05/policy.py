"""The policy MolmoSpaces actually builds and drives, for evaluation.

`pi05/model.py` is where the action path is defined and explained. This file is the
adapter that lets the simulator use it: MolmoSpaces has no gym API, it constructs a
policy class of its own and calls it one env step at a time, so our loop has to arrive as
a subclass rather than as a script.

    eval_main  --(module:Class)-->  Pi05EvalConfig
                                        |  policy_cls
                                        v
                                   Pi05EvalPolicy
                                        |  every env step
                                        v
                    obs_to_model_input -> inference_model -> model_output_to_action

What each step does:

    obs_to_model_input     pull the two camera images, the arm state and the instruction
                           out of the raw observation. Also stashes proprioception
                           (positions AND velocities), which the RL state needs and the
                           VLA does not see.
    inference_model        ask the VLA when the chunk is spent, optionally let RL Token
                           correct the chunk, then hand out one action with its arm
                           deltas resolved into absolute joint targets.
    model_output_to_action inherited: scales the gripper channel to the environment's
                           0..255 and packages the {"arm", "gripper"} dict.

The run's settings do not arrive as constructor arguments -- `run_evaluation` builds the
config class with no arguments, in a worker process. They arrive as a JSON file named by
$PI05_RUN_CONFIG, written by `pi05/eval.py`. See `pi05/config.py`.
"""

from __future__ import annotations

import logging

import numpy as np
from molmo_spaces.configs.policy_configs_baselines import PiPolicyConfig
from molmo_spaces.evaluation.configs.evaluation_configs import PiPolicyEvalConfig
from molmo_spaces.policy.base_policy import PolicyFactory
from molmo_spaces.policy.learned_policy.pi_policy import PI_Policy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad
from molmo_spaces.utils.function_utils import make_lenient
from pydantic import Field

from .action_space import GRIPPER_QPOS_OPEN, normalize_instruction
from .client import Pi05Client
from .config import ARM_DOF, load_run_config, needs_encoder
from .model import ChunkExecutor

log = logging.getLogger(__name__)

#: The geometry the training data was built with. The stock client resizes here too, so
#: keeping it client-side reproduces the measured path exactly.
IMAGE_SIZE = 224


class Pi05EvalPolicy(PI_Policy):
    """Frozen pi0.5 in the simulator, with an optional RL-Token correction."""

    def __init__(self, exp_config, task=None) -> None:
        # MolmoSpaces builds policies as policy_cls(exp_config, task), so `task` must own
        # the second position or a worker process would pass the task in as something else.
        super().__init__(exp_config)
        if task is not None:
            self.task = task

        self.run = load_run_config()
        problem = self.run.validate()
        if problem:
            raise ValueError(f"the run config is not runnable: {problem}")

        remote = exp_config.policy_config.remote_config or {}
        self.client = Pi05Client(
            remote.get("host", "127.0.0.1"),
            int(remote.get("port", 8080)),
        )
        self.executor = ChunkExecutor(self.run.chunk_size, self.run.conversion)

        # RL Token, only when an actor was named. Built lazily: the encoder wants a GPU
        # and a plain VLA benchmark must not pay for one.
        self.corrector = None
        # Tokens are fetched either because RL reads them, or because a corpus is being
        # collected. Collection wants every plan, not only the post-gate ones.
        self.recording = bool(getattr(self.run, "record_tokens", ""))
        self.wants_tokens = bool(getattr(self.run, "actor", "")) or self.recording

        self._proprio = np.zeros(2 * (ARM_DOF + 1), dtype=np.float32)
        self._instruction = ""
        self.step_index = 0
        #: Held False by the trainer during warmup, so the VLA drives while decisions are
        #: still recorded. Always True for evaluation.
        self.use_actor = True
        # PI_Policy.get_info reads this. prepare_model sets the real checkpoint
        # name; a default here stops eval from dying if get_history runs first
        # (new worker, SIGTERM between episodes).
        self.model_name = "pi05"
        self.reset()

    # --- lifecycle ------------------------------------------------------------------

    def prepare_model(self) -> None:
        """Connect to the frozen VLA. Overridden: we speak HTTP, not the websocket.

        One contract for evaluation and for RL training, so a benchmark number can never
        come from a different serving path than the run it is compared against.
        """
        info = self.client.wait_until_ready()
        self.model = self.client
        self.model_name = str(info.get("checkpoint", "pi05"))
        log.info("frozen pi0.5 at %s (%s)", self.client.url, self.model_name)

        # Tokens are wanted for two unrelated reasons and only one of them needs a
        # corrector: an RL run reads them through the encoder, a collection run merely
        # records them. Building one whenever tokens are wanted made every collection
        # episode die in torch.load on an empty encoder path, and because the runner
        # swallows per-episode exceptions the corpus stayed empty in silence.
        if needs_encoder(self.run) and self.corrector is None:
            self.corrector = self.build_corrector()
            if self.corrector is not None:
                log.info("RL Token active: %s", self.corrector.describe())

    def build_corrector(self):
        """The RL correction, loaded from the checkpoint the config names.

        Overridden during training, where the actor is live and being updated rather
        than loaded from disk.
        """
        from .rl_token import corrector_from_config

        return corrector_from_config(self.run)

    def want_tokens_now(self, step: int) -> bool:
        """Tokens cost ~7.9 MB per call, so they are only fetched when something reads
        them. With a latched gate that means from `gate_step` onwards -- unless a corpus
        is being collected, which wants the whole episode."""
        if self.recording:
            return True
        if not self.wants_tokens or self.corrector is None:
            return self.wants_tokens
        return self.corrector.gate.is_open(step)

    def on_plan(self, out: dict, executed_chunk, arm_state) -> None:
        """Called each time a new chunk is planned. Training records here."""

    def reset(self) -> None:
        super().reset()
        self.executor.reset()
        self.step_index = 0
        if self.corrector is not None:
            self.corrector.reset()

    # --- per step -------------------------------------------------------------------

    def obs_to_model_input(self, obs) -> dict:
        """What the VLA sees, plus the proprioception only the RL state uses."""
        if isinstance(obs, list | tuple):
            obs = obs[0]

        # The benchmark randomises the shoulder camera under a different key.
        exo_key = (
            "droid_shoulder_light_randomization"
            if "droid_shoulder_light_randomization" in obs
            else "exo_camera_1"
        )
        wrist_key = "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"

        arm_pos = np.asarray(obs["qpos"]["arm"][:ARM_DOF], dtype=np.float32)
        grip_pos = float(np.clip(obs["qpos"]["gripper"][0] / GRIPPER_QPOS_OPEN, 0.0, 1.0))
        arm_vel = np.asarray(obs["qvel"]["arm"][:ARM_DOF], dtype=np.float32)
        grip_vel = float(obs["qvel"]["gripper"][0]) / GRIPPER_QPOS_OPEN
        self._proprio = np.concatenate(
            [arm_pos, [grip_pos], arm_vel, [grip_vel]]
        ).astype(np.float32)

        self._instruction = normalize_instruction(self.task.get_task_description())
        return {
            "external_cam": resize_with_pad(obs[exo_key], IMAGE_SIZE, IMAGE_SIZE),
            "wrist_cam": resize_with_pad(obs[wrist_key], IMAGE_SIZE, IMAGE_SIZE),
            "instruction": self._instruction,
            "state": np.concatenate([arm_pos, [grip_pos]]).astype(np.float32),
        }

    def inference_model(self, model_input) -> np.ndarray:
        """One action, arm already absolute. The base class scales the gripper.

        Re-planning happens only when the chunk is spent, which is `chunk_size` env steps
        after the last plan.
        """
        if self.model is None:
            self.prepare_model()

        current_arm = np.asarray(model_input["state"], dtype=np.float32)[:ARM_DOF]

        if self.executor.needs_new_chunk:
            out = self.client.act(
                model_input["external_cam"],
                model_input["wrist_cam"],
                model_input["instruction"],
                model_input["state"],
                want_tokens=self.want_tokens_now(self.step_index),
            )
            chunk = out["actions"]
            if self.corrector is not None:
                chunk = self.corrector.correct(
                    chunk=chunk,
                    tokens=out.get("tokens"),
                    mask=out.get("mask"),
                    step=self.step_index,
                    proprio=self._proprio,
                    arm_state=current_arm,
                    use_actor=self.use_actor,
                )
            self.on_plan(out, chunk, current_arm)
            self.executor.load(chunk, current_arm)

        action = self.executor.next_action(current_arm)
        self.step_index += 1
        # Kept in sync so the base class's reporting hooks stay truthful.
        self.current_buffer_index = self.executor.index
        return action

    # --- reporting ------------------------------------------------------------------

    def get_info(self) -> dict:
        if not hasattr(self, "model_name"):
            self.model_name = "pi05"
        info = super().get_info()
        info["pi05_chunk_size"] = self.run.chunk_size
        info["pi05_conversion"] = self.run.conversion
        info["pi05_actor"] = getattr(self.run, "actor", "")
        if self.corrector is not None:
            info["pi05_gate_step"] = self.run.resolved_gate_step()
            info["pi05_gate_frac"] = getattr(self.run, "gate_frac", 0.0)
        return info


class Pi05RLPolicy(Pi05EvalPolicy):
    """The training-time policy: the same action path, plus recording.

    MolmoSpaces has no gym API -- an episode is a `run_evaluation` call that drives this
    object one env step at a time -- so the policy is also the recorder. It keeps what
    the episode produced and hands it over as an `rlt.replay.Rollout` afterwards.

    Per plan (every `chunk_size` env steps, which is also the decision point):

        ask the VLA           -> reference chunk, and the tokens that become z_rl
        gate closed or warmup -> commit the VLA's chunk
        gate open             -> commit the actor's sample instead
        always                -> record <x, a_ref, a> so the learner can close a row

    Only one stride is supported: one decision per plan. At a shorter stride a stored row
    would splice two committed chunks together and pair the result with a reference the
    robot never executed, which is a different algorithm, not a tuning knob.
    """

    def __init__(self, exp_config, task=None, *, corrector=None, on_decision=None) -> None:
        self._injected_corrector = corrector
        super().__init__(exp_config, task)
        self.wants_tokens = True
        self.corrector = corrector
        # Called after every decision, which is where Algorithm 1 puts the G learner
        # iterations. Left None for pure collection.
        self.on_decision = on_decision
        self.use_actor = False  # the trainer flips this on after warmup
        self.fatal_error: BaseException | None = None
        self.reset()

    def build_corrector(self):
        return self._injected_corrector

    def want_tokens_now(self, step: int) -> bool:
        """Before the gate the VLA drives, so tokens are only worth fetching if those
        transitions are going to be stored, or if the server is writing an AE corpus."""
        if getattr(self.run, "record_tokens", ""):
            return True
        if getattr(self.run, "store_pre_gate", False):
            return True
        return self.corrector is None or self.corrector.gate.is_open(step)

    @property
    def step(self) -> int:
        """Env steps taken this episode. Named `step` because rlt's EpisodeRunner
        reads it to tell an empty episode from a real one."""
        return self.step_index

    def reset(self) -> None:
        super().reset()
        self.decisions = []
        self.committed = []

    # --- recording ------------------------------------------------------------------

    def on_plan(self, out: dict, executed_chunk, arm_state) -> None:
        """Store the decision and the chunk that will actually run."""
        from rlt.replay import Decision

        corrector = self.corrector
        if corrector is None or corrector.last_state is None:
            # No tokens were fetched for this plan, so there is no RL state and this
            # stretch contributes nothing to the buffer. The actions still execute.
            self.committed.extend(
                np.asarray(executed_chunk, dtype=np.float32)[: self.run.chunk_size]
            )
            return

        # The buffer's action and reference must live in the same space, so what was
        # committed is expressed the same way the reference was.
        committed = corrector.reference_from_chunk(executed_chunk, arm_state)
        tokens = mask = None
        if getattr(self.run, "store_decision_tokens", False):
            tokens = corrector.last_tokens
            mask = corrector.last_mask
        self.decisions.append(
            Decision(
                step=self.step_index,
                state=corrector.last_state,
                reference=corrector.last_reference,
                tokens=tokens,
                mask=mask,
                log_prob=getattr(corrector, "last_log_prob", None),
            )
        )
        self.committed.extend(committed.reshape(self.run.chunk_size, -1))

    def inference_model(self, model_input) -> np.ndarray:
        try:
            action = super().inference_model(model_input)
        except BaseException as error:  # noqa: BLE001
            # The rollout runner swallows exceptions and calls the episode failed, which
            # would quietly poison the buffer. Keep it and re-raise outside.
            self.fatal_error = error
            raise
        if self.executor.index == 1 and self.on_decision is not None:
            self.on_decision(self)
        return action

    # --- draining -------------------------------------------------------------------

    def rollout_view(self):
        """The episode so far, with no terminal information: success ends the episode,
        so every window that closes mid-episode carries no reward."""
        from rlt.replay import Rollout

        return Rollout(
            steps=self.step_index,
            decisions=self.decisions,
            committed=self.committed,
            rewards=np.zeros(self.step_index, dtype=np.float32),
        )

    def pop_rollout(self, success: bool):
        """Drain the recorded episode. Rewards are sparse: +1 on the success step."""
        from rlt.replay import Rollout

        steps = self.step_index
        flags = self._success_flags(steps)
        terminal_step = int(np.argmax(flags)) if flags.any() else None
        if bool(flags.any()) != bool(success):
            raise RuntimeError(
                f"the runner reported success={success} but the task's success cache says "
                f"{bool(flags.any())} over {steps} steps"
            )
        rewards = np.zeros(steps, dtype=np.float32)
        if terminal_step is not None:
            rewards[terminal_step] = 1.0

        rollout = Rollout(
            steps=steps,
            decisions=self.decisions,
            committed=self.committed,
            rewards=rewards,
            terminal_step=terminal_step,
            success=bool(success),
        )
        self.reset()
        return rollout

    def _success_flags(self, n_steps: int) -> np.ndarray:
        """Per-step success from the task cache. Entry 0 belongs to task.reset."""
        cache = getattr(self.task, "success_cache", None)
        if cache is None:
            raise RuntimeError("the active task has no success_cache")
        if len(cache) < n_steps + 1:
            raise RuntimeError(
                f"success cache holds {len(cache) - 1} steps, the policy took {n_steps}"
            )
        return np.array([bool(np.asarray(v).reshape(-1)[0]) for v in cache[1 : n_steps + 1]])


# --------------------------------------------------------------------------------------
# The config eval_main is pointed at
# --------------------------------------------------------------------------------------


class Pi05PolicyConfig(PiPolicyConfig):
    """Same Franka, same cameras, same control period -- a different policy class."""

    policy_cls: type = Pi05EvalPolicy
    policy_factory: PolicyFactory | None = None

    # The base class re-plans when its own counter passes chunk_size. Our policy owns
    # chunking, so this is only reporting; it is set from the run config for honesty.
    chunk_size: int = 1
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    remote_config: dict | None = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_cls = Pi05EvalPolicy
        self.policy_factory = make_lenient(Pi05EvalPolicy)

        run = load_run_config()
        self.chunk_size = run.chunk_size
        self.grasping_type = run.grasping
        self.grasping_threshold = run.grasp_threshold
        # Host and port come from the run config so the client and the server can never
        # disagree about which endpoint this run uses.
        self.remote_config = {"host": "127.0.0.1", "port": int(run.port)}


class Pi05EvalConfig(PiPolicyEvalConfig):
    """Point eval_main here:  pi05.policy:Pi05EvalConfig"""

    policy_config: Pi05PolicyConfig = Field(default_factory=Pi05PolicyConfig)
