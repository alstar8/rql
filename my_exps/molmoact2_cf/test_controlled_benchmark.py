"""CPU/schema tests for the controlled house-0 kettle benchmark generator."""

from __future__ import annotations

import ast
import copy
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_controlled_benchmark import (
    BenchmarkValidationError,
    DEFAULT_N_TRAIN,
    DEFAULT_N_VAL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_ROBOT_COUNT_PER_SPLIT,
    DEFAULT_TRAIN_KETTLE_COUNT,
    DEFAULT_VAL_KETTLE_COUNT,
    GOAL_Z_OFFSET_M,
    HOUSE_INDEX,
    INSTRUCTION,
    KETTLE_XY_DEDUP_M,
    KETTLE_YAW_DEDUP_RAD,
    MASTER_SEED,
    POSE_ATOL,
    ROBOT_XY_DEDUP_M,
    ROBOT_YAW_DEDUP_RAD,
    SCENE_DATASET,
    TARGET_OBJECT,
    TASK_CLASS,
    _assert_episode_contract,
    build_generation_config,
    build_pair_layout,
    derive_seed,
    find_anchor_benchmark,
    load_verified_anchor,
    pose_hash,
    pose_is_distinct,
    required_axis_counts,
    robot_state_hash,
    stable_hash,
    validate_output,
    wrapped_angle_distance,
    yaw_from_pose,
)
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickDroidMiniBench,
)
from molmo_spaces.evaluation.benchmark_schema import (
    EpisodeSpec,
    PickTaskSpec,
    load_all_episodes,
)
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler


def _yaw_pose(x: float, y: float, yaw_rad: float) -> list[float]:
    return [
        x,
        y,
        1.0,
        math.cos(yaw_rad / 2.0),
        0.0,
        0.0,
        math.sin(yaw_rad / 2.0),
    ]


@pytest.fixture(scope="module")
def anchor_dir() -> Path:
    try:
        return find_anchor_benchmark()
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def anchor(anchor_dir: Path) -> EpisodeSpec:
    return load_verified_anchor(anchor_dir)


def test_default_factorizations_are_complete_cartesian_products() -> None:
    train = build_pair_layout(
        DEFAULT_N_TRAIN,
        DEFAULT_TRAIN_KETTLE_COUNT,
        DEFAULT_ROBOT_COUNT_PER_SPLIT,
    )
    val = build_pair_layout(
        DEFAULT_N_VAL,
        DEFAULT_VAL_KETTLE_COUNT,
        DEFAULT_ROBOT_COUNT_PER_SPLIT,
    )

    assert len(train) == 24
    assert len(val) == 12
    assert required_axis_counts(train) == (6, 4)
    assert required_axis_counts(val) == (3, 4)
    assert {(pair.kettle_index, pair.robot_index) for pair in train} == {
        (kettle_index, robot_index)
        for kettle_index in range(6)
        for robot_index in range(4)
    }
    assert {(pair.kettle_index, pair.robot_index) for pair in val} == {
        (kettle_index, robot_index)
        for kettle_index in range(3)
        for robot_index in range(4)
    }


@pytest.mark.parametrize(
    ("n_episodes", "expected_axes"),
    [
        (1, (1, 1)),
        (2, (1, 2)),
        (4, (1, 4)),
        (5, (2, 4)),
    ],
)
def test_smoke_counts_use_deterministic_grid_prefixes(
    n_episodes: int,
    expected_axes: tuple[int, int],
) -> None:
    first = build_pair_layout(n_episodes, 6, 4)
    second = build_pair_layout(n_episodes, 6, 4)
    assert first == second
    assert required_axis_counts(first) == expected_axes


@pytest.mark.parametrize("n_episodes", [0, -1, 25])
def test_invalid_train_counts_are_rejected(n_episodes: int) -> None:
    with pytest.raises(ValueError):
        build_pair_layout(n_episodes, 6, 4)


def test_seed_derivation_is_stable_and_stage_separated() -> None:
    first = derive_seed(MASTER_SEED, "episode", "train", 0, 0)
    assert first == derive_seed(MASTER_SEED, "episode", "train", 0, 0)
    assert first != derive_seed(MASTER_SEED, "episode", "val", 0, 0)
    assert first != derive_seed(MASTER_SEED, "episode", "train", 0, 1)
    assert 0 <= first < 2**31 - 1


def test_pose_hash_rounding_and_robot_state_hash() -> None:
    pose = _yaw_pose(1.0, 2.0, math.radians(20.0))
    near_pose = copy.deepcopy(pose)
    near_pose[0] += 1e-10
    assert pose_hash(pose) == pose_hash(near_pose)

    qpos = {"arm": [0.0] * 7, "gripper": [0.00296, 0.00296]}
    assert robot_state_hash(pose, qpos) == stable_hash(
        {"robot_base_pose": pose, "robot_init_qpos": qpos}
    )


def test_pose_dedup_requires_xy_and_yaw_separation() -> None:
    anchor_pose = _yaw_pose(0.0, 0.0, 0.0)

    too_close_xy = _yaw_pose(KETTLE_XY_DEDUP_M / 2.0, 0.0, math.radians(30.0))
    distinct, reason = pose_is_distinct(
        too_close_xy,
        [anchor_pose],
        KETTLE_XY_DEDUP_M,
        KETTLE_YAW_DEDUP_RAD,
    )
    assert not distinct
    assert reason is not None and reason.startswith("xy_dedup")

    too_close_yaw = _yaw_pose(KETTLE_XY_DEDUP_M * 2.0, 0.0, KETTLE_YAW_DEDUP_RAD / 2.0)
    distinct, reason = pose_is_distinct(
        too_close_yaw,
        [anchor_pose],
        KETTLE_XY_DEDUP_M,
        KETTLE_YAW_DEDUP_RAD,
    )
    assert not distinct
    assert reason is not None and reason.startswith("yaw_dedup")

    valid = _yaw_pose(
        KETTLE_XY_DEDUP_M * 2.0,
        0.0,
        KETTLE_YAW_DEDUP_RAD * 2.0,
    )
    assert pose_is_distinct(
        valid,
        [anchor_pose],
        KETTLE_XY_DEDUP_M,
        KETTLE_YAW_DEDUP_RAD,
    ) == (True, None)


def test_wrapped_yaw_distance_handles_pi_boundary() -> None:
    distance = wrapped_angle_distance(math.radians(179.0), math.radians(-179.0))
    assert math.isclose(distance, math.radians(2.0), abs_tol=1e-12)
    assert math.isclose(yaw_from_pose(_yaw_pose(0.0, 0.0, distance)), distance)


def test_robot_split_leakage_thresholds_are_enforceable() -> None:
    train_pose = _yaw_pose(0.0, 0.0, 0.0)
    leaked_val_pose = _yaw_pose(
        ROBOT_XY_DEDUP_M * 2.0,
        0.0,
        ROBOT_YAW_DEDUP_RAD / 2.0,
    )
    distinct, reason = pose_is_distinct(
        leaked_val_pose,
        [train_pose],
        ROBOT_XY_DEDUP_M,
        ROBOT_YAW_DEDUP_RAD,
    )
    assert not distinct
    assert reason is not None and reason.startswith("yaw_dedup")


def test_anchor_episode_zero_is_exact_verified_template(
    anchor: EpisodeSpec,
) -> None:
    assert anchor.house_index == HOUSE_INDEX
    assert anchor.scene_dataset == SCENE_DATASET
    assert anchor.get_task_cls() == TASK_CLASS
    assert anchor.task["pickup_obj_name"] == TARGET_OBJECT
    assert anchor.language.task_description == INSTRUCTION
    assert TARGET_OBJECT in anchor.scene_modifications.object_poses
    PickTaskSpec.model_validate(anchor.task)


def test_anchor_schema_and_controlled_contract_pass(anchor: EpisodeSpec) -> None:
    anchor_dict = anchor.model_dump()
    _assert_episode_contract(anchor_dict, anchor)
    parsed = EpisodeSpec.model_validate(anchor_dict)
    expected_goal = copy.deepcopy(parsed.task["pickup_obj_start_pose"])
    expected_goal[2] += GOAL_Z_OFFSET_M
    np.testing.assert_allclose(
        parsed.task["pickup_obj_goal_pose"],
        expected_goal,
        atol=POSE_ATOL,
        rtol=0.0,
    )


def test_generation_config_pins_anchor_without_creating_simulator(
    anchor: EpisodeSpec,
    tmp_path: Path,
) -> None:
    config = build_generation_config(anchor, MASTER_SEED, tmp_path)
    assert config.seed == MASTER_SEED
    assert config.scene_dataset == anchor.scene_dataset
    assert config.data_split == anchor.data_split
    assert config.task_sampler_config.house_inds == [HOUSE_INDEX]
    assert config.task_config.pickup_obj_name == TARGET_OBJECT
    assert config.robot_config.init_qpos["arm"] == anchor.robot.init_qpos["arm"]
    assert len(config.camera_config.cameras) == len(anchor.cameras)


def test_contract_rejects_wrong_goal(anchor: EpisodeSpec) -> None:
    episode = anchor.model_dump()
    episode["task"]["pickup_obj_goal_pose"][2] += 0.01
    with pytest.raises(BenchmarkValidationError, match="goal pose"):
        _assert_episode_contract(episode, anchor)


def test_contract_rejects_non_kettle_scene_change(anchor: EpisodeSpec) -> None:
    episode = anchor.model_dump()
    other_object = next(
        name
        for name in episode["scene_modifications"]["object_poses"]
        if name != TARGET_OBJECT
    )
    episode["scene_modifications"]["object_poses"][other_object][0] += 0.001
    with pytest.raises(BenchmarkValidationError, match="non-kettle poses changed"):
        _assert_episode_contract(episode, anchor)


def test_generator_and_tests_have_no_inline_imports() -> None:
    files = (Path(__file__).resolve().with_name("generate_controlled_benchmark.py"), Path(__file__))
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inline_imports = [
                    child
                    for child in ast.walk(node)
                    if isinstance(child, (ast.Import, ast.ImportFrom))
                ]
                assert not inline_imports, f"Inline import found in {path}: {inline_imports}"


def test_existing_outputs_validate_without_simulation(anchor_dir: Path) -> None:
    if not (DEFAULT_OUTPUT_ROOT / "manifest.json").is_file():
        pytest.skip("Controlled benchmark artifacts have not been generated yet")
    summary = validate_output(DEFAULT_OUTPUT_ROOT, anchor_dir=anchor_dir)
    assert summary["validated"] is True
    assert summary["load_all_episodes_compatible"] is True


@pytest.mark.integration
def test_json_sampler_replays_one_generated_episode() -> None:
    """Optional one-episode JSON sampler replay; no policy inference is run."""

    if os.environ.get("MOLMOSPACES_RUN_INTEGRATION") != "1":
        pytest.skip("Set MOLMOSPACES_RUN_INTEGRATION=1 to run JSON sampler replay")
    output_root = Path(
        os.environ.get("CONTROLLED_BENCHMARK_ROOT", str(DEFAULT_OUTPUT_ROOT))
    ).resolve()
    if not (output_root / "train" / "benchmark.json").is_file():
        pytest.skip(f"Generated train benchmark not found at {output_root}")

    validate_output(output_root)
    episode = load_all_episodes(output_root / "train")[0]
    config = FrankaPickDroidMiniBench(seed=episode.seed or MASTER_SEED)
    config.profile = False
    config.profiler = None
    sampler = JsonEvalTaskSampler(config, episode)
    task = None
    try:
        task = sampler.sample_task(house_index=HOUSE_INDEX)
        assert task is not None
        assert task.get_task_description() == INSTRUCTION
        assert task.config.task_config.pickup_obj_name == TARGET_OBJECT
        object_manager = task.env.object_managers[task.env.current_batch_index]
        kettle = object_manager.get_object_by_name(TARGET_OBJECT)
        np.testing.assert_allclose(
            kettle.position,
            episode.task["pickup_obj_start_pose"][:3],
            atol=POSE_ATOL,
            rtol=0.0,
        )
    finally:
        if task is not None:
            task.close()
        sampler.close()
