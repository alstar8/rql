"""Unit tests for V16 always-collect decision logic (no GPU)."""

from __future__ import annotations


def decide_use_actor(
    *,
    deploy_actor: bool,
    actor_mode: str,
    always_collect_actor: bool,
    guide_on_reference: bool,
    eval_force_reference: bool,
    collect_episode_index: int,
    actor_bc_episodes: int,
    actor_mixture_prob: float,
    mixture_draw: bool,
) -> tuple[bool, str]:
    """Mirror of train_rlt_online RLTOnlinePolicy collect branching."""
    use_actor = bool(deploy_actor and actor_mode == "rlt")
    policy = "reference"
    if (
        not use_actor
        and always_collect_actor
        and actor_mode == "rlt"
        and not guide_on_reference
        and not eval_force_reference
        and collect_episode_index >= actor_bc_episodes
    ):
        return True, "actor"
    if (
        not use_actor
        and actor_mode == "rlt"
        and actor_mixture_prob > 0.0
        and not eval_force_reference
        and mixture_draw
    ):
        return True, "mixture_actor"
    return use_actor, policy


def test_paper_collect_after_warmup() -> None:
    use, name = decide_use_actor(
        deploy_actor=False,
        actor_mode="rlt",
        always_collect_actor=True,
        guide_on_reference=False,
        eval_force_reference=False,
        collect_episode_index=50,
        actor_bc_episodes=50,
        actor_mixture_prob=0.0,
        mixture_draw=False,
    )
    assert use and name == "actor"


def test_warmup_stays_on_reference() -> None:
    use, name = decide_use_actor(
        deploy_actor=False,
        actor_mode="rlt",
        always_collect_actor=True,
        guide_on_reference=False,
        eval_force_reference=False,
        collect_episode_index=10,
        actor_bc_episodes=50,
        actor_mixture_prob=0.0,
        mixture_draw=False,
    )
    assert not use and name == "reference"


def test_guide_on_reference_never_paper_collects() -> None:
    use, name = decide_use_actor(
        deploy_actor=False,
        actor_mode="rlt",
        always_collect_actor=True,
        guide_on_reference=True,
        eval_force_reference=False,
        collect_episode_index=100,
        actor_bc_episodes=50,
        actor_mixture_prob=0.0,
        mixture_draw=False,
    )
    assert not use


if __name__ == "__main__":
    test_paper_collect_after_warmup()
    test_warmup_stays_on_reference()
    test_guide_on_reference_never_paper_collects()
    print("ok")
