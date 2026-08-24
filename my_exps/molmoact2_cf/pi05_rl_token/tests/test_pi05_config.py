"""The config is the interface, so its guard rails are worth pinning.

Every rule here exists because breaking it produced a number that looked fine and was
not: a run at the wrong chunk size, an actor evaluated in a space it was not trained in,
or a benchmark that silently skipped half its episodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi05.config import (  # noqa: E402
    SCENES,
    SELECTED_SCENES,
    VLA_CHUNK,
    EvalConfig,
    RLConfig,
    apply_overrides,
    describe,
    load_run_config,
    needs_encoder,
    save_run_config,
)


# --- scenes ------------------------------------------------------------------------------


def test_every_scene_carries_a_normalised_task():
    for name, scene in SCENES.items():
        assert scene.task == scene.task.lower(), name
        assert scene.task.endswith("."), name


def test_a_scene_that_trains_and_measures_on_the_same_episodes_says_so():
    """A jittered scene keeps two disjoint halves. A multi-task set deliberately does not:
    the question there is whether one actor helps across the tasks it was trained on, so
    the same 20 episodes are both the training pool and what gets reported. That is fine,
    but it must be stated in the scene, not discovered from a path."""
    for name, scene in SCENES.items():
        if scene.benchmark_train == scene.benchmark_eval:
            assert "MULTI-TASK" in scene.note, f"{name}: shares its benchmarks silently"


def test_the_selected_scenes_exist_and_say_what_was_measured():
    """The previous selection became unreproducible because nothing recorded why those
    scenes were chosen. A selected scene has to carry its measurement in its note."""
    for name in SELECTED_SCENES:
        assert name in SCENES, name
        note = SCENES[name].note
        assert "probed" in note, f"{name}: note does not say what was measured"
        assert "band" in note.lower(), f"{name}: note does not say which band"


def test_the_inherited_scenes_are_not_selected_by_default():
    """house2/house10/house21 came from a profiling run recorded as contaminated, and
    their rates were measured under chunk 1 + plan_time. They stay in SCENES so old runs
    can be reproduced, but nothing should pick them up without being asked to."""
    for name in ("house2", "house10", "house21"):
        assert name not in SELECTED_SCENES


def test_an_unknown_scene_fails_loudly():
    with pytest.raises(KeyError, match="unknown scene"):
        EvalConfig(scene="kitchen").validate()


def test_evaluation_defaults_to_the_held_out_half():
    """Reporting a rate measured on the episodes RL trained on is the mistake this split
    exists to prevent, so the safe half has to be the default."""
    cfg = EvalConfig(scene="house10")
    assert cfg.split == "eval"
    assert str(cfg.benchmark_dir()).endswith("bench_house10_eval48")


def test_the_training_half_has_to_be_asked_for_and_only_to_collect():
    assert "not a result" in EvalConfig(scene="house10", split="train").validate()
    ok = EvalConfig(scene="house10", split="train", record_tokens="/tmp/corpus")
    assert ok.validate() == ""
    assert str(ok.benchmark_dir()).endswith("bench_house10_train48")


def test_an_unknown_split_is_refused():
    assert "split must be" in EvalConfig(scene="house10", split="holdout").validate()


def test_rl_training_can_only_see_the_training_half():
    """RLConfig has no split field on purpose: the trainer is never given the option."""
    assert str(RLConfig(scene="house21").benchmark_dir()).endswith("bench_house21_train48")
    assert "split" not in RLConfig.__dataclass_fields__


def test_the_run_tag_records_which_half_ran():
    assert "_eval_" in EvalConfig(scene="house2").default_tag()
    assert "_train_" in EvalConfig(
        scene="house2", split="train", record_tokens="/tmp/c"
    ).default_tag()


def test_single_task_scenes_keep_two_different_halves():
    for name, scene in SCENES.items():
        if "MULTI-TASK" in scene.note:
            continue
        assert scene.benchmark_train != scene.benchmark_eval, name


def test_an_empty_scene_means_the_shipped_val_benchmark():
    cfg = EvalConfig(scene="")
    assert cfg.validate() == ""
    assert "FrankaPickDroidMiniBench" in str(cfg.benchmark_dir())


def test_describe_mentions_every_scene():
    text = describe()
    for name in SCENES:
        assert name in text


# --- the execution regime -----------------------------------------------------------------


def test_defaults_are_runnable():
    assert EvalConfig().validate() == ""
    assert RLConfig().validate() == ""


@pytest.mark.parametrize("bad", [0, -1, VLA_CHUNK + 1])
def test_chunk_size_must_fit_the_prediction(bad):
    assert "chunk_size" in EvalConfig(chunk_size=bad).validate()
    assert "chunk_size" in RLConfig(chunk_size=bad).validate()


def test_an_unknown_conversion_is_rejected():
    with pytest.raises(ValueError, match="plan_time"):
        EvalConfig(conversion="later").validate()


def test_more_than_one_worker_is_refused():
    """Two clients on one pi0.5 server time out during the handshake and their episodes
    are skipped silently, which once reported 7 of 16 episodes as the whole run."""
    assert "workers must be 1" in EvalConfig(workers=4).validate()


def test_the_run_tag_names_the_regime():
    """Two runs are only comparable if chunk_size and conversion agree, so both belong in
    the directory name rather than only inside result.json."""
    tag = EvalConfig(scene="house10", chunk_size=8, conversion="plan_time").default_tag()
    assert "house10" in tag and "chunk8" in tag and "plan_time" in tag
    assert "step_time" in EvalConfig(conversion="step_time").default_tag()


def test_the_defaults_are_the_regime_the_val_sweep_settled_on():
    """chunk 8 + step_time scored 62.5% on the held-out 128 episodes against 40.6% for
    the old chunk 1 + plan_time default (Fisher p=0.0007). Changing these defaults means
    new runs stop being comparable with that sweep, so it should be deliberate."""
    cfg = EvalConfig()
    assert (cfg.chunk_size, cfg.conversion) == (8, "step_time")
    assert RLConfig().rl_action_space == "delta"


# --- RL ------------------------------------------------------------------------------------


def test_stride_defaults_to_the_chunk_size():
    assert RLConfig(chunk_size=4, stride=0).resolved_stride() == 4


def test_stride_must_divide_the_chunk():
    assert "must divide" in RLConfig(chunk_size=4, stride=3).validate()


def test_warmup_must_end_before_the_run_does():
    assert "warmup" in RLConfig(episodes=10, warmup_episodes=10).validate()


def test_warmup_zero_is_allowed():
    """Online can start with the actor in control from episode 0 after AC pretrain."""
    assert RLConfig(episodes=10, warmup_episodes=0).validate() == ""


def test_absolute_actions_require_plan_time_conversion():
    """The corrector subtracts the arm state so the executor can add it back. Only
    plan_time adds back the same state, so any other pairing shifts the actor's intent."""
    problem = RLConfig(rl_action_space="absolute", conversion="step_time").validate()
    assert "plan_time" in problem


def test_delta_actions_are_fine_with_step_time():
    assert RLConfig(rl_action_space="delta", conversion="step_time").validate() == ""


def test_an_actor_without_its_encoder_is_refused():
    assert "token_ae" in EvalConfig(actor="/tmp/agent.pt").validate()


def test_a_negative_gate_step_below_catalog_sentinel_is_refused():
    assert "gate_step" in RLConfig(gate_step=-5).validate()


def test_gate_step_zero_means_rl_from_the_first_env_step():
    """Default: no frozen-VLA prefix. The scene catalog (mug 56, …) is not applied."""
    from pi05.config import SCENES

    assert RLConfig().gate_step == 0
    assert RLConfig(scene="desk_mug").resolved_gate_step() == 0
    assert EvalConfig(scene="desk_mug", actor="/tmp/a.pt", token_ae="/tmp/e.pt").resolved_gate_step() == 0
    assert SCENES["desk_mug"].gate_step == 56  # catalog still there for gate_step=-1


def test_gate_step_n_is_a_vla_prefix_of_n_steps():
    assert RLConfig(scene="desk_mug", gate_step=56).resolved_gate_step() == 56
    assert (
        EvalConfig(scene="desk_mug", actor="/tmp/a.pt", token_ae="/tmp/e.pt", gate_step=56).resolved_gate_step()
        == 56
    )


def test_gate_step_minus_one_uses_the_scene_catalog():
    from pi05.config import SCENES

    assert RLConfig(scene="desk_mug", gate_step=-1).resolved_gate_step() == SCENES["desk_mug"].gate_step


def test_an_unknown_encoder_lists_what_exists():
    with pytest.raises((FileNotFoundError, OSError)):
        RLConfig(encoder="nonexistent-encoder").token_ae_path()


# --- overrides and round-tripping -----------------------------------------------------------


def test_overrides_are_coerced_to_the_field_type():
    cfg = EvalConfig()
    apply_overrides(cfg, ["chunk_size=8", "save_video=false", "grasp_threshold=0.25"])
    assert cfg.chunk_size == 8 and cfg.chunk_size.__class__ is int
    assert cfg.save_video is False
    assert cfg.grasp_threshold == pytest.approx(0.25)


def test_an_unknown_override_names_the_known_fields():
    with pytest.raises(KeyError, match="chunk_size"):
        apply_overrides(EvalConfig(), ["chunksize=8"])


def test_a_malformed_override_is_refused():
    with pytest.raises(ValueError, match="name=value"):
        apply_overrides(EvalConfig(), ["chunk_size"])


def test_a_config_survives_the_trip_to_the_worker_process(tmp_path):
    """The worker reads the run config from JSON, so a field that does not survive the
    round trip is a setting the simulator silently ignores."""
    cfg = EvalConfig(scene="house10", chunk_size=8, conversion="step_time", episodes=12)
    path = save_run_config(cfg, tmp_path / "config.json")
    back = load_run_config(path)
    assert isinstance(back, EvalConfig)
    assert (back.scene, back.chunk_size, back.conversion, back.episodes) == (
        "house10", 8, "step_time", 12,
    )


def test_an_rl_config_round_trips_as_an_rl_config(tmp_path):
    path = save_run_config(RLConfig(scene="house2", gate_step=17), tmp_path / "c.json")
    back = load_run_config(path)
    assert isinstance(back, RLConfig)
    assert back.gate_step == 17


def test_a_missing_run_config_env_says_which_variable(monkeypatch):
    monkeypatch.delenv("PI05_RUN_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="PI05_RUN_CONFIG"):
        load_run_config()


def test_each_selected_scene_has_a_gate_on_a_chunk_boundary():
    """The gate was read off contact sheets per scene -- when the gripper arrives depends
    on where the object is, so it does not transfer. A gate inside a chunk would silently
    slip to the next boundary, because decisions only happen there."""
    for name in SELECTED_SCENES:
        gate = SCENES[name].gate_step
        assert gate > 0, f"{name}: no gate chosen"
        assert gate % EvalConfig().chunk_size == 0, f"{name}: gate {gate} is mid-chunk"


def test_training_with_catalog_gate_on_a_scene_without_one_is_refused():
    """gate_step=-1 asks for the measured catalog. Inherited scenes have none."""
    problem = RLConfig(scene="house21", gate_step=-1).validate()
    assert "gate_step" in problem


def test_training_from_step_zero_is_fine_on_a_scene_without_a_catalog_gate():
    problem = RLConfig(scene="house21").validate()
    assert "gate" not in (problem or "")


def test_a_mid_chunk_gate_override_is_refused():
    assert "mid-chunk" not in RLConfig(gate_step=56).validate()
    assert "inside a chunk" in RLConfig(gate_step=50).validate()


def test_a_collection_run_wants_tokens_but_not_an_encoder():
    """Recording the corpus and refining a chunk both request tokens, but only the second
    loads the phase-1 encoder. Conflating them emptied a whole collection in silence."""
    collecting = EvalConfig(scene="desk_mug", split="train", record_tokens="/tmp/corpus")
    assert collecting.record_tokens
    assert not needs_encoder(collecting)

    evaluating_actor = EvalConfig(actor="/tmp/agent.pt", token_ae="/tmp/ae.pt")
    assert needs_encoder(evaluating_actor)

    plain = EvalConfig()
    assert not needs_encoder(plain)


def test_every_selected_scene_has_an_encoder_and_a_combined_one_exists():
    """The encoder is the RL state, so it must come from the corpus of the regime being
    run. The superseded directory holds an `ae_combined.pt` too, which is the one stale
    file that would have been loaded without complaint."""
    from pi05.config import TOKEN_AE_DIR

    assert TOKEN_AE_DIR.name.endswith("step_time"), TOKEN_AE_DIR
    for name in [*SELECTED_SCENES, "combined"]:
        assert (TOKEN_AE_DIR / f"ae_{name}.pt").exists(), f"missing encoder for {name}"


def test_an_actor_is_evaluated_at_the_gate_it_was_trained_at():
    """Train and eval share the same default (step 0) and the same explicit N."""
    for name in SELECTED_SCENES:
        train_gate = RLConfig(scene=name, encoder=name).resolved_gate_step()
        eval_gate = EvalConfig(
            scene=name, actor="/tmp/a.pt", token_ae="/tmp/e.pt"
        ).resolved_gate_step()
        assert train_gate == eval_gate == 0, name
        assert (
            RLConfig(scene=name, encoder=name, gate_step=56).resolved_gate_step()
            == EvalConfig(
                scene=name, actor="/tmp/a.pt", token_ae="/tmp/e.pt", gate_step=56
            ).resolved_gate_step()
            == 56
        )


def test_evaluating_an_actor_with_catalog_gate_on_a_scene_without_one_is_refused():
    problem = EvalConfig(
        scene="house21", actor="/tmp/a.pt", token_ae="/tmp/e.pt", gate_step=-1
    ).validate()
    assert "gate_step" in problem


def test_beta_is_scaled_for_the_delta_action_space():
    """The paper's beta=1.0 assumes a reference of ~0.7 rad (absolute joint targets). We
    refine deltas of ~0.009 rad and the penalty goes as the square of that, so 1.0 leaves
    the actor free to ignore the plan: measured deviation was 197% of the reference and
    the motion was 7.5x rougher. At 100 it is 67% and 3.5x, with a higher success rate."""
    assert RLConfig().beta == 100.0
    assert RLConfig().rl_action_space == "delta"
