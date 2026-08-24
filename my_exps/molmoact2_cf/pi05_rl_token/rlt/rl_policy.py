"""The MolmoSpaces policy that runs Algorithm 1's rollout half.

MolmoSpaces has no gym API: an episode is a `run_evaluation` call that drives
this object one env step at a time. So the policy is also the recorder -- it
keeps what the episode produced and hands it over as a `Rollout` afterwards.

Per env step t:

    t % stride == 0   ask the VLA -> reference chunk + z_rl, record a decision
    t % C == 0        commit the next C actions: the VLA's chunk during warmup,
                      the actor's sample afterwards
    always            execute committed[t]

stride divides C, so every chunk boundary is also a decision point.
"""

from __future__ import annotations

import numpy as np
from molmo_spaces.policy.learned_policy.molmoact2_policy import MolmoAct2_Policy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad

from .config import ACTION_DIM, CHUNK, GRIPPER_SCALE, PROPRIO_DIM
from .replay import Decision, Rollout


def point_config_at(policy_cfg, vla) -> None:
    """Make the config name the server the client already points at.

    `MolmoAct2_Policy.__init__` health-checks the address in the config and
    refuses to start if nothing answers. Collectors each get their own client,
    so without this every collector except the one matching the config default
    dies at startup.
    """
    policy_cfg.remote_config = {"host": vla.host, "port": vla.port}


def build_vla(policy_cfg):
    from .vla import VlaClient

    remote = policy_cfg.remote_config or {}
    client = VlaClient(remote.get("host", "localhost"), int(remote.get("port", 8000)))
    client.wait_until_ready()
    return client


def build_encoder(policy_cfg):
    from .vla import TokenEncoder

    if not getattr(policy_cfg, "token_ae", ""):
        raise ValueError("no token AE checkpoint: pass one, or set RLT_TOKEN_AE for worker processes")
    return TokenEncoder(policy_cfg.token_ae, getattr(policy_cfg, "device", "cuda:0"))


def build_agent(policy_cfg, encoder):
    """Only when a checkpoint is named; otherwise the policy is the plain VLA."""
    import torch

    from .agent import RLTokenAgent
    from .config import OnlineConfig

    path = getattr(policy_cfg, "actor", "")
    if not path:
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OnlineConfig(**{k: v for k, v in ckpt["config"].items() if k in OnlineConfig.__dataclass_fields__})
    cfg.device = getattr(policy_cfg, "device", "cuda:0")
    agent = RLTokenAgent(cfg, encoder.z_dim + PROPRIO_DIM, CHUNK * ACTION_DIM, CHUNK)
    agent.load(path)
    return agent


class RLTokenPolicy(MolmoAct2_Policy):
    """Frozen VLA + (optionally) the RL actor on top of it."""

    def __init__(
        self,
        exp_config,
        task=None,
        *,
        vla=None,
        encoder=None,
        agent=None,
        chunk: int = CHUNK,
        stride: int = 2,
        explore: bool = True,
        on_decision=None,
    ) -> None:
        # MolmoSpaces builds policies as policy_cls(exp_config, task), so `task`
        # must own the second position -- everything else is keyword-only, or a
        # worker process would quietly pass the task in as the VLA client.
        exp_config.policy_config.chunk_size = chunk
        if vla is not None:
            point_config_at(exp_config.policy_config, vla)
        super().__init__(exp_config)
        if task is not None:
            self.task = task
        # A worker process started by run_evaluation gets nothing but the config,
        # so build the missing pieces from it.
        policy_cfg = exp_config.policy_config
        self.vla = vla if vla is not None else build_vla(policy_cfg)
        self.encoder = encoder if encoder is not None else build_encoder(policy_cfg)
        self.agent = agent if agent is not None else build_agent(policy_cfg, self.encoder)
        self.chunk = chunk
        self.stride = stride
        self.explore = explore
        self.use_actor = self.agent is not None  # flipped off during warmup by the trainer
        # Called after every decision point, which is where Algorithm 1 puts the
        # G learner iterations. Left None for pure collection or evaluation.
        self.on_decision = on_decision
        self.fatal_error: BaseException | None = None
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.decisions: list[Decision] = []
        self.committed: list[np.ndarray] = []
        self.actions_buffer = None  # kept in sync for the base class's state hooks
        self.current_buffer_index = 0

    # --- per-step -----------------------------------------------------------

    def obs_to_model_input(self, obs) -> dict:
        """The VLA sees positions only; the RL state also sees velocities."""
        if isinstance(obs, list | tuple):
            obs = obs[0]
        exo_key = (
            "droid_shoulder_light_randomization"
            if "droid_shoulder_light_randomization" in obs
            else "exo_camera_1"
        )
        wrist_key = "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"

        arm_pos = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip_pos = np.clip(float(obs["qpos"]["gripper"][0]) / GRIPPER_SCALE, 0.0, 1.0)
        arm_vel = np.asarray(obs["qvel"]["arm"][:7], dtype=np.float32)
        grip_vel = float(obs["qvel"]["gripper"][0]) / GRIPPER_SCALE

        return {
            "external_cam": resize_with_pad(obs[exo_key], self.image_size, self.image_size),
            "wrist_cam": resize_with_pad(obs[wrist_key], self.image_size, self.image_size),
            "instruction": self.task.get_task_description().lower(),
            "state": np.concatenate([arm_pos, [grip_pos]]).astype(np.float32),
            "proprio": np.concatenate([arm_pos, [grip_pos], arm_vel, [grip_vel]]).astype(np.float32),
        }

    def inference_model(self, model_input) -> np.ndarray:
        try:
            return self._inference(model_input)
        except BaseException as error:  # noqa: BLE001
            # The rollout runner swallows exceptions and calls the episode failed,
            # which would quietly poison the buffer. Keep it and re-raise outside.
            self.fatal_error = error
            raise

    def _inference(self, model_input) -> np.ndarray:
        t = self.step
        decided = t % self.stride == 0

        if decided:
            out = self.vla.act(
                model_input["external_cam"],
                model_input["wrist_cam"],
                model_input["instruction"],
                model_input["state"],
            )
            reference = out["actions"][: self.chunk].reshape(-1)
            z_rl = self.encoder.encode(out["tokens"], out["mask"])
            state = np.concatenate([z_rl, model_input["proprio"]]).astype(np.float32)
            self.decisions.append(Decision(step=t, state=state, reference=reference))

        if t % self.chunk == 0:
            decision = self.decisions[-1]
            if self.use_actor:
                chunk = self.agent.act(decision.state, decision.reference, explore=self.explore)
            else:
                chunk = decision.reference
            self.committed.extend(np.asarray(chunk, np.float32).reshape(self.chunk, ACTION_DIM))

        self.step += 1
        self.current_buffer_index = self.step % self.chunk
        action = self.committed[t]

        if decided and self.on_decision is not None:
            self.on_decision(self)
        return action

    # --- after the episode --------------------------------------------------

    def rollout_view(self) -> Rollout:
        """The episode so far. No terminal information: success ends the episode,
        so every window that closes mid-episode carries no reward."""
        return Rollout(
            steps=self.step,
            decisions=self.decisions,
            committed=self.committed,
            rewards=np.zeros(self.step, dtype=np.float32),
        )

    def pop_rollout(self, success: bool) -> Rollout:
        """Drain the recorded episode. Rewards are sparse: +1 on the success step."""
        n = self.step
        flags = self._success_flags(n)
        terminal_step = int(np.argmax(flags)) if flags.any() else None
        if bool(flags.any()) != bool(success):
            raise RuntimeError(
                f"the runner reported success={success} but the task's success cache "
                f"says {bool(flags.any())} over {n} steps"
            )

        rewards = np.zeros(n, dtype=np.float32)
        if terminal_step is not None:
            rewards[terminal_step] = 1.0

        rollout = Rollout(
            steps=n,
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
            raise RuntimeError(f"success cache holds {len(cache) - 1} steps, the policy took {n_steps}")
        return np.array([bool(np.asarray(v).reshape(-1)[0]) for v in cache[1 : n_steps + 1]])
