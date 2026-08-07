"""Frozen-expert residual ConsensusFlow.

Restore a pretrained flow-matching actor, freeze it, and train a small
guidance residual with critic distillation plus direct residual RL.
"""

from agents.dflrql9 import DFLRQL9Agent, get_config as get_v9_config


class DFLRQL10Agent(DFLRQL9Agent):
    """Train G as an RL residual policy around a frozen flow expert."""


def get_config():
    config = get_v9_config()
    config.agent_name = "dflrql10"
    # The restored behavior expert is immutable.
    config.alpha = 0.0
    config.actor_q_coef = 0.0
    config.actor_lookahead_use_guidance = False
    config.freeze_actor = True
    # Keep some consensus distill for critic agreement, but let residual RL
    # dominate once a return-maximizing G signal exists.
    config.distill_coef = 0.25
    config.guidance_q_coef = 0.0
    config.guidance_rollout_q_coef = 1.0
    config.guidance_energy_coef = 0.01
    # Maximize improvement over the frozen expert, not absolute Q.
    config.guidance_use_advantage = True
    # Allow anti-BC residual updates during RL; keep safety at inference.
    config.guidance_rl_bypass_safety = True
    return config
