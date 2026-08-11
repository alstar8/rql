"""Generate and validate the controlled ProcTHOR house-0 kettle benchmark.

The default output is a 6 x 4 train Cartesian product and a disjoint 3 x 4
validation Cartesian product. Kettle poses are sampled with
``place_object_near`` on the kettle's anchor support. Robot states are sampled
with ``CPUMujocoEnv.place_robot_near``. Every Cartesian pair is replayed and
checked for robot collision and non-colliding grasps before it is frozen.

Generation intentionally does not run a policy rollout or MolmoAct2 inference.
Use ``--validate-only`` for a simulation-free schema and invariant check.
"""

from __future__ import annotations

import argparse
import base64
import copy
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

_HERE = Path(__file__).resolve().parent
_MOLMOSPACES_ROOT = Path(__file__).resolve().parents[3] / "molmospaces"
if str(_MOLMOSPACES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOLMOSPACES_ROOT))

from molmo_spaces.configs.camera_configs import CameraSystemConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickDroidMiniBench,
)
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.evaluation.benchmark_schema import (
    EpisodeSpec,
    PickTaskSpec,
    load_all_episodes,
)
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.tasks.task_sampler_errors import RobotPlacementError
from molmo_spaces.tasks.json_eval_task_sampler import camera_spec_to_config
from molmo_spaces.utils.grasp_sample import get_noncolliding_grasp_mask
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom, place_object_near
from molmo_spaces.utils.pose import pose_mat_to_7d, pos_quat_to_pose_mat

_CONVERTER_PATH = (
    _MOLMOSPACES_ROOT / "scripts" / "benchmarks" / "create_json_benchmark.py"
)
_CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "molmospaces_create_json_benchmark",
    _CONVERTER_PATH,
)
if _CONVERTER_SPEC is None or _CONVERTER_SPEC.loader is None:
    raise ImportError(f"Cannot load benchmark converter from {_CONVERTER_PATH}")
_CONVERTER_MODULE = importlib.util.module_from_spec(_CONVERTER_SPEC)
_CONVERTER_SPEC.loader.exec_module(_CONVERTER_MODULE)
frozen_config_to_episode_spec = _CONVERTER_MODULE.frozen_config_to_episode_spec


BENCHMARK_ID = "house0_kettle_v13"
MASTER_SEED = 20260811
HOUSE_INDEX = 0
SCENE_DATASET = "procthor-10k"
TASK_CLASS = "molmo_spaces.tasks.pick_task.PickTask"
TARGET_OBJECT = "boiler_335af387651d9869b84abf1fb099bf69_1_0_2"
INSTRUCTION = "pick up the kettle."
GOAL_Z_OFFSET_M = 0.05

DEFAULT_TRAIN_KETTLE_COUNT = 6
DEFAULT_VAL_KETTLE_COUNT = 3
DEFAULT_ROBOT_COUNT_PER_SPLIT = 4
DEFAULT_N_TRAIN = DEFAULT_TRAIN_KETTLE_COUNT * DEFAULT_ROBOT_COUNT_PER_SPLIT
DEFAULT_N_VAL = DEFAULT_VAL_KETTLE_COUNT * DEFAULT_ROBOT_COUNT_PER_SPLIT

# A pose is accepted only when BOTH its XY distance and wrapped yaw distance
# meet these thresholds against every already accepted pose. This stronger
# rule makes each axis independently deduplicated and makes leakage checks
# unambiguous.
KETTLE_XY_DEDUP_M = 0.045
KETTLE_YAW_DEDUP_RAD = math.radians(10.0)
ROBOT_XY_DEDUP_M = 0.12
ROBOT_YAW_DEDUP_RAD = math.radians(10.0)

KETTLE_PLACEMENT_MIN_RADIUS_M = 0.025
KETTLE_PLACEMENT_MAX_RADIUS_M = 0.18
KETTLE_PLACEMENT_ATTEMPTS = 200
ROBOT_SAMPLE_RADIUS_RANGE_M = (0.38, 0.50)
ROBOT_REPLAY_MAX_DISTANCE_M = 0.70
MAX_CANDIDATE_ATTEMPTS = 160
POSE_ATOL = 1e-7
POSE_HASH_DECIMALS = 8

DEFAULT_OUTPUT_ROOT = _HERE / "runs" / "benchmarks" / BENCHMARK_ID
_ANCHOR_SUFFIX = (
    Path("benchmarks")
    / "molmospaces-bench-v1"
    / "procthor-10k"
    / "FrankaPickDroidMiniBench"
    / "FrankaPickDroidMiniBench_json_benchmark_20251231"
)
DEFAULT_ANCHOR_CANDIDATES = (
    Path.home() / ".cache" / "molmospaces" / "assets" / _ANCHOR_SUFFIX,
    Path.home()
    / ".cache"
    / "molmo-spaces-resources"
    / "benchmarks"
    / "molmospaces-bench-v1"
    / "20260408"
    / "procthor-10k"
    / "FrankaPickDroidMiniBench"
    / "FrankaPickDroidMiniBench_json_benchmark_20251231",
)

RequestMode = Literal["sample_kettle", "sample_robot", "replay_pair"]


class BenchmarkValidationError(ValueError):
    """Raised when a generated benchmark violates its controlled contract."""


@dataclass(frozen=True)
class PairIndex:
    """One index pair in a split's requested factorization."""

    kettle_index: int
    robot_index: int

    @property
    def pair_id(self) -> str:
        return f"k{self.kettle_index:02d}_r{self.robot_index:02d}"


@dataclass
class SamplingRequest:
    """State requested from the controlled PickTaskSampler."""

    mode: RequestMode
    kettle_pose: list[float] | None = None
    robot_base_pose: list[float] | None = None
    robot_init_qpos: dict[str, list[float]] | None = None


@dataclass
class SampleOutcome:
    """Copied state from one successful simulator-backed sample."""

    kettle_pose: list[float]
    robot_base_pose: list[float]
    robot_init_qpos: dict[str, list[float]]
    support: dict[str, Any]
    feasible_grasp_count: int
    episode: dict[str, Any] | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonicalize(value: Any, decimals: int = POSE_HASH_DECIMALS) -> Any:
    value = _jsonable(value)
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, dict):
        return {
            key: _canonicalize(value[key], decimals=decimals)
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [_canonicalize(item, decimals=decimals) for item in value]
    return value


def stable_hash(value: Any, decimals: int = POSE_HASH_DECIMALS) -> str:
    """Hash JSON-like data after stable sorting and float rounding."""

    payload = json.dumps(
        _canonicalize(value, decimals=decimals),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pose_hash(pose: list[float]) -> str:
    return stable_hash(pose)


def robot_state_hash(
    robot_base_pose: list[float],
    robot_init_qpos: dict[str, list[float]],
) -> str:
    return stable_hash(
        {
            "robot_base_pose": robot_base_pose,
            "robot_init_qpos": robot_init_qpos,
        }
    )


def derive_seed(master_seed: int, *labels: object) -> int:
    """Derive a stable positive 31-bit seed without relying on Python hash()."""

    text = ":".join([str(master_seed), *(str(label) for label in labels)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def yaw_from_pose(pose: list[float]) -> float:
    """Return wrapped Z yaw for scalar-first [x, y, z, qw, qx, qy, qz]."""

    if len(pose) != 7:
        raise ValueError(f"Expected seven-element pose, got {len(pose)}")
    qw, qx, qy, qz = pose[3:]
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)


def wrapped_angle_distance(angle_a: float, angle_b: float) -> float:
    return abs((angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi)


def pose_separation(pose_a: list[float], pose_b: list[float]) -> tuple[float, float]:
    xy_distance = float(np.linalg.norm(np.asarray(pose_a[:2]) - np.asarray(pose_b[:2])))
    yaw_distance = wrapped_angle_distance(yaw_from_pose(pose_a), yaw_from_pose(pose_b))
    return xy_distance, yaw_distance


def pose_is_distinct(
    candidate: list[float],
    accepted: list[list[float]],
    xy_threshold_m: float,
    yaw_threshold_rad: float,
) -> tuple[bool, str | None]:
    """Require independent XY and yaw separation from every accepted pose."""

    for index, existing in enumerate(accepted):
        xy_distance, yaw_distance = pose_separation(candidate, existing)
        if xy_distance < xy_threshold_m:
            return (
                False,
                f"xy_dedup: candidate is {xy_distance:.6f}m from pose {index}; "
                f"minimum is {xy_threshold_m:.6f}m",
            )
        if yaw_distance < yaw_threshold_rad:
            return (
                False,
                f"yaw_dedup: candidate is {math.degrees(yaw_distance):.6f}deg "
                f"from pose {index}; minimum is {math.degrees(yaw_threshold_rad):.6f}deg",
            )
    return True, None


def build_pair_layout(
    n_episodes: int,
    max_kettle_count: int,
    max_robot_count: int,
) -> list[PairIndex]:
    """Build a deterministic full grid or prefix for small smoke generation."""

    max_episodes = max_kettle_count * max_robot_count
    if not 1 <= n_episodes <= max_episodes:
        raise ValueError(f"n_episodes must be in [1, {max_episodes}], got {n_episodes}")
    full_grid = [
        PairIndex(kettle_index, robot_index)
        for kettle_index, robot_index in itertools.product(
            range(max_kettle_count), range(max_robot_count)
        )
    ]
    return full_grid[:n_episodes]


def required_axis_counts(layout: list[PairIndex]) -> tuple[int, int]:
    return (
        max(pair.kettle_index for pair in layout) + 1,
        max(pair.robot_index for pair in layout) + 1,
    )


def find_anchor_benchmark(anchor_override: Path | None = None) -> Path:
    candidates = (anchor_override,) if anchor_override is not None else DEFAULT_ANCHOR_CANDIDATES
    for candidate in candidates:
        if candidate is not None and (candidate / "benchmark.json").is_file():
            return candidate.resolve()
    rendered = "\n".join(f"  - {candidate}" for candidate in candidates if candidate is not None)
    raise FileNotFoundError(
        "Default FrankaPickDroidMiniBench anchor was not found. Checked:\n" + rendered
    )


def load_verified_anchor(anchor_dir: Path) -> EpisodeSpec:
    episodes = load_all_episodes(anchor_dir)
    if not episodes:
        raise BenchmarkValidationError(f"No episodes in anchor benchmark: {anchor_dir}")
    anchor = episodes[0]
    failures = []
    if anchor.house_index != HOUSE_INDEX:
        failures.append(f"house_index={anchor.house_index}, expected {HOUSE_INDEX}")
    if anchor.scene_dataset != SCENE_DATASET:
        failures.append(f"scene_dataset={anchor.scene_dataset!r}, expected {SCENE_DATASET!r}")
    if anchor.get_task_cls() != TASK_CLASS:
        failures.append(f"task_cls={anchor.get_task_cls()!r}, expected {TASK_CLASS!r}")
    if anchor.task.get("pickup_obj_name") != TARGET_OBJECT:
        failures.append(
            f"pickup_obj_name={anchor.task.get('pickup_obj_name')!r}, expected {TARGET_OBJECT!r}"
        )
    if anchor.language.task_description != INSTRUCTION:
        failures.append(
            f"instruction={anchor.language.task_description!r}, expected {INSTRUCTION!r}"
        )
    anchor_pose = anchor.scene_modifications.object_poses.get(TARGET_OBJECT)
    if anchor_pose is None:
        failures.append(f"anchor object_poses does not contain {TARGET_OBJECT}")
    if failures:
        raise BenchmarkValidationError("Anchor episode 0 contract failed: " + "; ".join(failures))
    PickTaskSpec.model_validate(anchor.task)
    return anchor


def _capture_robot_qpos(robot_view: Any) -> dict[str, list[float]]:
    qpos = robot_view.get_qpos_dict(robot_view.move_group_ids())
    return {
        group_name: np.asarray(group_qpos, dtype=float).tolist()
        for group_name, group_qpos in qpos.items()
    }


def _source_git_revision(repository: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(repository)}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        result.update({"revision": revision, "dirty": dirty})
    except (OSError, subprocess.CalledProcessError) as error:
        result["unavailable_reason"] = str(error)
    return result


def active_v12_processes() -> list[str]:
    """Return active V12 trainer/server commands; never mutate process state."""

    try:
        output = subprocess.run(
            ["ps", "-eo", "args="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    v12_server_ports = tuple(range(8600, 8624)) + tuple(range(9060, 9074))
    active = []
    for line in output.splitlines():
        is_trainer = (
            "train_rlt_online.py" in line and "rlt_cf_v12_shortlist" in line
        )
        is_launcher = "launch_v12_shortlist" in line
        is_server = "molmoact2_cf/serve.py" in line and any(
            f"--port {port}" in line for port in v12_server_ports
        )
        if is_trainer or is_launcher or is_server:
            active.append(line.strip())
    return active


class ControlledHouse0KettleSampler(PickTaskSampler):
    """PickTaskSampler that samples API-backed factors and replays their cross product."""

    def __init__(self, config: FrankaPickDroidMiniBench, anchor: EpisodeSpec) -> None:
        self.anchor = anchor
        self.anchor_object_poses = copy.deepcopy(anchor.scene_modifications.object_poses)
        self.anchor_kettle_pose = copy.deepcopy(self.anchor_object_poses[TARGET_OBJECT])
        self.anchor_init_qpos = {
            key: copy.deepcopy(value)
            for key, value in anchor.robot.init_qpos.items()
            if len(value) > 0
        }
        self.base_init_qpos_noise_range = copy.deepcopy(
            config.robot_config.init_qpos_noise_range
        )
        self.request = SamplingRequest(mode="sample_kettle")
        self.last_support: dict[str, Any] = {}
        self.last_feasible_grasp_count = 0
        self.last_robot_collision_free = False
        super().__init__(config)

    def configure(self, request: SamplingRequest) -> None:
        self.request = request

    def _restore_anchor_objects(self, env: Any) -> None:
        for object_name, pose in self.anchor_object_poses.items():
            body = create_mlspaces_body(env.current_data, object_name)
            body.position = np.asarray(pose[:3], dtype=float)
            body.quat = np.asarray(pose[3:], dtype=float)
        mujoco.mj_forward(env.current_model, env.current_data)

    def _support_metadata(self, env: Any, support_geom_id: int | None) -> dict[str, Any]:
        if support_geom_id is None:
            return {"available": False, "unavailable_reason": "get_supporting_geom returned None"}
        model = env.current_model
        data = env.current_data
        geom_name = model.geom(support_geom_id).name
        body_id = int(model.geom_bodyid[support_geom_id])
        body_name = model.body(body_id).name
        metadata: dict[str, Any] = {
            "available": True,
            "geom_id": int(support_geom_id),
            "geom_name": geom_name,
            "body_id": body_id,
            "body_name": body_name,
        }
        try:
            center, size = geom_aabb(model, data, [support_geom_id])
            metadata["aabb_center"] = center.tolist()
            metadata["aabb_size"] = size.tolist()
        except (RuntimeError, ValueError) as error:
            metadata["aabb_unavailable_reason"] = str(error)
        return metadata

    def randomize_scene(self, env: Any, robot_view: Any) -> None:
        if self.request.robot_init_qpos is None:
            self.config.robot_config.init_qpos = copy.deepcopy(self.anchor_init_qpos)
            self.config.robot_config.init_qpos_noise_range = copy.deepcopy(
                self.base_init_qpos_noise_range
            )
        else:
            self.config.robot_config.init_qpos = copy.deepcopy(
                self.request.robot_init_qpos
            )
            self.config.robot_config.init_qpos_noise_range = None

        super().randomize_scene(env, robot_view)
        self._restore_anchor_objects(env)

        target_body = create_mlspaces_body(env.current_data, TARGET_OBJECT)
        target_id = target_body.body_id
        anchor_support_id = get_supporting_geom(env.current_data, target_id)
        if anchor_support_id is None:
            raise RuntimeError("Anchor kettle has no supporting geometry in house 0")

        if self.request.mode == "sample_kettle":
            place_object_near(
                data=env.current_data,
                object_id=target_id,
                placement_point=np.asarray(self.anchor_kettle_pose[:3], dtype=float),
                min_dist=KETTLE_PLACEMENT_MIN_RADIUS_M,
                max_dist=KETTLE_PLACEMENT_MAX_RADIUS_M,
                max_tries=KETTLE_PLACEMENT_ATTEMPTS,
                supporting_geom_id=anchor_support_id,
                z_eps=0.0,
            )
        elif self.request.kettle_pose is None:
            raise ValueError(f"{self.request.mode} requires kettle_pose")
        else:
            target_body.position = np.asarray(self.request.kettle_pose[:3], dtype=float)
            target_body.quat = np.asarray(self.request.kettle_pose[3:], dtype=float)

        mujoco.mj_forward(env.current_model, env.current_data)
        actual_support_id = get_supporting_geom(env.current_data, target_id)
        if actual_support_id != anchor_support_id:
            raise RuntimeError(
                "Kettle is not supported by its anchor geometry: "
                f"anchor={anchor_support_id}, actual={actual_support_id}"
            )
        self.last_support = self._support_metadata(env, actual_support_id)

        self.config.task_config.pickup_obj_name = TARGET_OBJECT
        scene_candidates = self._get_scene_objects(env)
        target_candidates = [
            candidate for candidate in scene_candidates if candidate.name == TARGET_OBJECT
        ]
        if len(target_candidates) != 1:
            raise RuntimeError(f"Target kettle {TARGET_OBJECT} is not a valid pick candidate")
        self.candidate_objects = target_candidates

    def _sample_and_place_robot(self, env: Any) -> None:
        if self.request.mode != "replay_pair":
            super()._sample_and_place_robot(env)
            self.last_robot_collision_free = not env.check_robot_collision_in_current_pose(
                "robot_0/"
            )
            if not self.last_robot_collision_free:
                raise RobotPlacementError("place_robot_near returned a colliding robot state")
            return

        if self.request.robot_base_pose is None or self.request.robot_init_qpos is None:
            raise ValueError("replay_pair requires robot_base_pose and robot_init_qpos")

        task_config = self.config.task_config
        object_manager = env.object_managers[env.current_batch_index]
        pickup_obj = object_manager.get_object_by_name(TARGET_OBJECT)
        start_pose = pose_mat_to_7d(pickup_obj.pose)
        robot_view = env.current_robot.robot_view
        robot_view.base.pose = pos_quat_to_pose_mat(
            self.request.robot_base_pose[:3],
            self.request.robot_base_pose[3:],
        )
        mujoco.mj_forward(env.current_model, env.current_data)

        self.last_robot_collision_free = not env.check_robot_collision_in_current_pose(
            "robot_0/"
        )
        if not self.last_robot_collision_free:
            raise RobotPlacementError("Replayed robot state collides with the scene")

        robot_xy = np.asarray(self.request.robot_base_pose[:2], dtype=float)
        kettle_xy = np.asarray(start_pose[:2], dtype=float)
        robot_kettle_distance = float(np.linalg.norm(robot_xy - kettle_xy))
        if not 0.0 < robot_kettle_distance < ROBOT_REPLAY_MAX_DISTANCE_M:
            raise RobotPlacementError(
                f"Replayed robot distance {robot_kettle_distance:.6f}m is outside "
                f"(0.000000, {ROBOT_REPLAY_MAX_DISTANCE_M:.6f})"
            )

        for controller in env.current_robot.controllers.values():
            controller.reset()
        env.current_robot.set_stationary()
        env.current_robot.compute_control()

        task_config.pickup_obj_start_pose = start_pose.tolist()
        task_config.robot_base_pose = copy.deepcopy(self.request.robot_base_pose)
        goal_pose = start_pose.copy()
        goal_pose[2] += GOAL_Z_OFFSET_M
        task_config.pickup_obj_goal_pose = goal_pose.tolist()

    def _strict_feasible_grasp_count(self, env: Any) -> int:
        object_manager = env.object_managers[env.current_batch_index]
        pickup_obj = object_manager.get_object_by_name(TARGET_OBJECT)
        grasps = get_pickup_grasps(
            env,
            pickup_obj,
            include_flipped=False,
            grasp_libraries=self.config.task_sampler_config.grasp_libraries,
        )
        if len(grasps) == 0:
            raise RuntimeError("No pickup grasps returned for target kettle")
        mask = get_noncolliding_grasp_mask(
            env.current_model,
            env.current_data,
            grasps,
            64,
        )
        count = int(np.sum(mask))
        if count <= 0:
            raise RuntimeError(
                f"Target kettle has 0/{len(grasps)} non-colliding grasps"
            )
        return count

    def _sample_task(self, env: Any) -> PickTask:
        task = super()._sample_task(env)
        self.last_feasible_grasp_count = self._strict_feasible_grasp_count(env)
        return task


def build_generation_config(
    anchor: EpisodeSpec,
    master_seed: int,
    output_root: Path,
) -> FrankaPickDroidMiniBench:
    """Build the default MiniBench config while pinning episode-0 template state."""

    config = FrankaPickDroidMiniBench(seed=master_seed)
    config.profile = False
    config.profiler = None
    config.output_dir = output_root
    config.scene_dataset = anchor.scene_dataset
    config.data_split = anchor.data_split
    config.robot_config.init_qpos = {
        key: copy.deepcopy(value)
        for key, value in anchor.robot.init_qpos.items()
        if len(value) > 0
    }
    config.camera_config = CameraSystemConfig(
        img_resolution=anchor.img_resolution,
        cameras=[camera_spec_to_config(camera) for camera in anchor.cameras],
    )
    config.task_config.pickup_obj_name = TARGET_OBJECT
    config.task_config.referral_expressions = {"pickup_obj_name": "kettle"}
    config.task_config.referral_expressions_priority = copy.deepcopy(
        anchor.language.referral_expressions_priority
    )
    sampler_config = config.task_sampler_config
    sampler_config.task_sampler_class = ControlledHouse0KettleSampler
    sampler_config.house_inds = [HOUSE_INDEX]
    sampler_config.samples_per_house = 100_000
    sampler_config.max_tasks = math.inf
    sampler_config.check_robot_placement_visibility = True
    sampler_config.base_pose_sampling_radius_range = ROBOT_SAMPLE_RADIUS_RANGE_M
    sampler_config.randomize_lighting = False
    sampler_config.randomize_textures = False
    sampler_config.randomize_textures_all = False
    sampler_config.randomize_robot_textures = False
    sampler_config.randomize_dynamics = False
    sampler_config.filter_for_grasps = True
    return config


def _decode_frozen_config(frozen_config: str) -> Any:
    return pickle.loads(base64.b64decode(frozen_config))


def _make_episode_from_task(
    task: PickTask,
    sampler: ControlledHouse0KettleSampler,
    anchor: EpisodeSpec,
    anchor_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Freeze through MolmoSpaces and convert through its JSON benchmark API."""

    observation = task.get_observations()
    frozen_b64 = task.config.freeze_task_config(observation, task=task)
    task.frozen_config = frozen_b64
    frozen_config = _decode_frozen_config(frozen_b64)
    source = anchor.source
    episode_spec = frozen_config_to_episode_spec(
        frozen_config=frozen_config,
        obs_scene={"task_description": INSTRUCTION},
        house_id=HOUSE_INDEX,
        scene_dataset=anchor.scene_dataset,
        data_split=anchor.data_split,
        source_h5_file=source.h5_file if source is not None else str(anchor_dir),
        source_traj_key=source.traj_key if source is not None else "episode_0",
        source_episode_length=source.episode_length if source is not None else 0,
        img_resolution=anchor.img_resolution,
        camera_system_class=source.camera_system_class if source is not None else None,
        source_data_date=source.source_data_date if source is not None else None,
        benchmark_created_date=datetime.date.today().isoformat(),
        task_horizon_sec=30,
    )
    episode_spec.seed = seed
    episode_spec.task["task_type"] = "pick"
    episode_spec.task_relevant_objects = [TARGET_OBJECT]
    episode_spec.language.task_description = INSTRUCTION
    episode_spec.language.referral_expressions = {"pickup_obj_name": "kettle"}
    episode = episode_spec.model_dump()
    episode["generation_checks"] = {
        "place_object_near": True,
        "place_robot_near_source_or_validated_replay": True,
        "robot_collision_free": sampler.last_robot_collision_free,
        "feasible_grasp_count": sampler.last_feasible_grasp_count,
        "freeze_task_config": True,
        "frozen_config_to_episode_spec": True,
    }
    return episode


def sample_once(
    sampler: ControlledHouse0KettleSampler,
    request: SamplingRequest,
    seed: int,
    anchor: EpisodeSpec,
    anchor_dir: Path,
    freeze_episode: bool,
) -> SampleOutcome:
    sampler.configure(request)
    sampler.seed_task_sampling(seed)
    task: PickTask | None = None
    try:
        sampled_task = sampler.sample_task(house_index=HOUSE_INDEX)
        if sampled_task is None:
            raise RuntimeError("PickTaskSampler returned None")
        if not isinstance(sampled_task, PickTask):
            raise TypeError(f"Expected PickTask, got {type(sampled_task).__name__}")
        task = sampled_task
        robot_view = task.env.current_robot.robot_view
        kettle_pose = copy.deepcopy(task.config.task_config.pickup_obj_start_pose)
        robot_base_pose = copy.deepcopy(task.config.task_config.robot_base_pose)
        robot_init_qpos = _capture_robot_qpos(robot_view)
        episode = (
            _make_episode_from_task(task, sampler, anchor, anchor_dir, seed)
            if freeze_episode
            else None
        )
        return SampleOutcome(
            kettle_pose=kettle_pose,
            robot_base_pose=robot_base_pose,
            robot_init_qpos=robot_init_qpos,
            support=copy.deepcopy(sampler.last_support),
            feasible_grasp_count=sampler.last_feasible_grasp_count,
            episode=episode,
        )
    finally:
        if task is not None:
            task.close()


def _attempt_log(
    stage: str,
    seed: int,
    accepted: bool,
    reason: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    record = {
        "stage": stage,
        "seed": seed,
        "accepted": accepted,
        **_jsonable(details),
    }
    if reason is not None:
        record["reason"] = reason
    return record


def _collect_kettle_candidates(
    sampler: ControlledHouse0KettleSampler,
    anchor: EpisodeSpec,
    anchor_dir: Path,
    master_seed: int,
    train_count: int,
    val_count: int,
    generation_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    total_count = train_count + val_count
    for candidate_index in range(total_count):
        split = "train" if candidate_index < train_count else "val"
        split_index = candidate_index if split == "train" else candidate_index - train_count
        candidate_id = f"{split}_k{split_index:02d}"
        for attempt in range(MAX_CANDIDATE_ATTEMPTS):
            seed = derive_seed(master_seed, "kettle", candidate_index, attempt)
            try:
                outcome = sample_once(
                    sampler,
                    SamplingRequest(mode="sample_kettle"),
                    seed,
                    anchor,
                    anchor_dir,
                    freeze_episode=False,
                )
                distinct, reason = pose_is_distinct(
                    outcome.kettle_pose,
                    [candidate["pose"] for candidate in accepted],
                    KETTLE_XY_DEDUP_M,
                    KETTLE_YAW_DEDUP_RAD,
                )
                if not distinct:
                    generation_log.append(
                        _attempt_log(
                            "kettle_candidate",
                            seed,
                            False,
                            reason,
                            candidate_id=candidate_id,
                            attempt=attempt,
                            pose=outcome.kettle_pose,
                        )
                    )
                    continue
                candidate = {
                    "id": candidate_id,
                    "split": split,
                    "index": split_index,
                    "seed": seed,
                    "pose": outcome.kettle_pose,
                    "pose_hash": pose_hash(outcome.kettle_pose),
                    "yaw_rad": yaw_from_pose(outcome.kettle_pose),
                    "supporting_geometry": outcome.support,
                    "feasible_grasp_count": outcome.feasible_grasp_count,
                }
                accepted.append(candidate)
                generation_log.append(
                    _attempt_log(
                        "kettle_candidate",
                        seed,
                        True,
                        candidate_id=candidate_id,
                        attempt=attempt,
                        pose=outcome.kettle_pose,
                        supporting_geometry=outcome.support,
                        feasible_grasp_count=outcome.feasible_grasp_count,
                    )
                )
                break
            except Exception as error:
                generation_log.append(
                    _attempt_log(
                        "kettle_candidate",
                        seed,
                        False,
                        f"{type(error).__name__}: {error}",
                        candidate_id=candidate_id,
                        attempt=attempt,
                    )
                )
        else:
            raise RuntimeError(
                f"Unable to generate {candidate_id} after {MAX_CANDIDATE_ATTEMPTS} attempts"
            )
    return accepted[:train_count], accepted[train_count:]


def _collect_robot_candidates(
    sampler: ControlledHouse0KettleSampler,
    anchor: EpisodeSpec,
    anchor_dir: Path,
    master_seed: int,
    train_count: int,
    val_count: int,
    kettle_candidates: list[dict[str, Any]],
    generation_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    total_count = train_count + val_count
    reference_kettle_pose = sampler.anchor_kettle_pose
    sampler.used_robot_positions[TARGET_OBJECT] = []

    for candidate_index in range(total_count):
        split = "train" if candidate_index < train_count else "val"
        split_index = candidate_index if split == "train" else candidate_index - train_count
        candidate_id = f"{split}_r{split_index:02d}"
        for attempt in range(MAX_CANDIDATE_ATTEMPTS):
            sampler.used_robot_positions[TARGET_OBJECT] = [
                np.asarray(candidate["base_pose"][:3], dtype=float)
                for candidate in accepted
            ]
            seed = derive_seed(master_seed, "robot", candidate_index, attempt)
            try:
                outcome = sample_once(
                    sampler,
                    SamplingRequest(
                        mode="sample_robot",
                        kettle_pose=reference_kettle_pose,
                    ),
                    seed,
                    anchor,
                    anchor_dir,
                    freeze_episode=False,
                )
                distinct, reason = pose_is_distinct(
                    outcome.robot_base_pose,
                    [candidate["base_pose"] for candidate in accepted],
                    ROBOT_XY_DEDUP_M,
                    ROBOT_YAW_DEDUP_RAD,
                )
                if not distinct:
                    generation_log.append(
                        _attempt_log(
                            "robot_candidate",
                            seed,
                            False,
                            reason,
                            candidate_id=candidate_id,
                            attempt=attempt,
                            base_pose=outcome.robot_base_pose,
                        )
                    )
                    continue

                init_qpos_hash = stable_hash(outcome.robot_init_qpos)
                if init_qpos_hash in {
                    candidate["init_qpos_hash"] for candidate in accepted
                }:
                    generation_log.append(
                        _attempt_log(
                            "robot_candidate",
                            seed,
                            False,
                            "init_qpos_dedup: identical robot joint initialization",
                            candidate_id=candidate_id,
                            attempt=attempt,
                        )
                    )
                    continue

                cross_validations = []
                for kettle in kettle_candidates:
                    replay_seed = derive_seed(
                        master_seed,
                        "robot_cross_validation",
                        candidate_index,
                        attempt,
                        kettle["id"],
                    )
                    replay = sample_once(
                        sampler,
                        SamplingRequest(
                            mode="replay_pair",
                            kettle_pose=kettle["pose"],
                            robot_base_pose=outcome.robot_base_pose,
                            robot_init_qpos=outcome.robot_init_qpos,
                        ),
                        replay_seed,
                        anchor,
                        anchor_dir,
                        freeze_episode=False,
                    )
                    cross_validations.append(
                        {
                            "kettle_id": kettle["id"],
                            "seed": replay_seed,
                            "feasible_grasp_count": replay.feasible_grasp_count,
                            "robot_collision_free": True,
                        }
                    )

                candidate = {
                    "id": candidate_id,
                    "split": split,
                    "index": split_index,
                    "seed": seed,
                    "base_pose": outcome.robot_base_pose,
                    "base_pose_hash": pose_hash(outcome.robot_base_pose),
                    "yaw_rad": yaw_from_pose(outcome.robot_base_pose),
                    "init_qpos": outcome.robot_init_qpos,
                    "init_qpos_hash": init_qpos_hash,
                    "robot_state_hash": robot_state_hash(
                        outcome.robot_base_pose,
                        outcome.robot_init_qpos,
                    ),
                    "cross_validations": cross_validations,
                }
                accepted.append(candidate)
                generation_log.append(
                    _attempt_log(
                        "robot_candidate",
                        seed,
                        True,
                        candidate_id=candidate_id,
                        attempt=attempt,
                        base_pose=outcome.robot_base_pose,
                        init_qpos=outcome.robot_init_qpos,
                        cross_validations=cross_validations,
                    )
                )
                break
            except Exception as error:
                generation_log.append(
                    _attempt_log(
                        "robot_candidate",
                        seed,
                        False,
                        f"{type(error).__name__}: {error}",
                        candidate_id=candidate_id,
                        attempt=attempt,
                    )
                )
        else:
            raise RuntimeError(
                f"Unable to generate {candidate_id} after {MAX_CANDIDATE_ATTEMPTS} attempts"
            )
    return accepted[:train_count], accepted[train_count:]


def _non_kettle_scene(
    object_poses: dict[str, list[float]],
) -> dict[str, list[float]]:
    return {
        object_name: pose
        for object_name, pose in object_poses.items()
        if object_name != TARGET_OBJECT
    }


def _assert_episode_contract(
    episode: dict[str, Any],
    anchor: EpisodeSpec,
) -> None:
    parsed = EpisodeSpec.model_validate(episode)
    PickTaskSpec.model_validate(parsed.task)
    errors = []
    if parsed.house_index != HOUSE_INDEX:
        errors.append(f"house_index={parsed.house_index}")
    if parsed.scene_dataset != SCENE_DATASET:
        errors.append(f"scene_dataset={parsed.scene_dataset!r}")
    if parsed.get_task_cls() != TASK_CLASS:
        errors.append(f"task_cls={parsed.get_task_cls()!r}")
    if parsed.task.get("pickup_obj_name") != TARGET_OBJECT:
        errors.append(f"pickup_obj_name={parsed.task.get('pickup_obj_name')!r}")
    if parsed.language.task_description != INSTRUCTION:
        errors.append(f"instruction={parsed.language.task_description!r}")

    start = parsed.task["pickup_obj_start_pose"]
    goal = parsed.task["pickup_obj_goal_pose"]
    expected_goal = copy.deepcopy(start)
    expected_goal[2] += GOAL_Z_OFFSET_M
    if not np.allclose(goal, expected_goal, atol=POSE_ATOL, rtol=0.0):
        errors.append("goal pose is not start + 0.05m Z")
    scene_start = parsed.scene_modifications.object_poses.get(TARGET_OBJECT)
    if scene_start is None or not np.allclose(
        start, scene_start, atol=POSE_ATOL, rtol=0.0
    ):
        errors.append("task start pose does not match scene target pose")

    anchor_non_kettle = _non_kettle_scene(anchor.scene_modifications.object_poses)
    episode_non_kettle = _non_kettle_scene(parsed.scene_modifications.object_poses)
    if set(anchor_non_kettle) != set(episode_non_kettle):
        errors.append("non-kettle mobile object set differs from anchor")
    else:
        changed = [
            object_name
            for object_name in anchor_non_kettle
            if not np.allclose(
                anchor_non_kettle[object_name],
                episode_non_kettle[object_name],
                atol=POSE_ATOL,
                rtol=0.0,
            )
        ]
        if changed:
            errors.append(f"non-kettle poses changed: {changed[:5]}")

    checks = episode.get("generation_checks", {})
    if checks and (
        not checks.get("robot_collision_free")
        or int(checks.get("feasible_grasp_count", 0)) <= 0
    ):
        errors.append("generation collision/grasp checks did not pass")
    if errors:
        raise BenchmarkValidationError("; ".join(errors))


def _build_split(
    split: str,
    layout: list[PairIndex],
    kettles: list[dict[str, Any]],
    robots: list[dict[str, Any]],
    sampler: ControlledHouse0KettleSampler,
    anchor: EpisodeSpec,
    anchor_dir: Path,
    master_seed: int,
    generation_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = []
    episode_manifest = []
    for episode_index, pair in enumerate(layout):
        kettle = kettles[pair.kettle_index]
        robot = robots[pair.robot_index]
        pair_id = f"{split}_{pair.pair_id}"
        seed = derive_seed(master_seed, "episode", split, pair.kettle_index, pair.robot_index)
        outcome = sample_once(
            sampler,
            SamplingRequest(
                mode="replay_pair",
                kettle_pose=kettle["pose"],
                robot_base_pose=robot["base_pose"],
                robot_init_qpos=robot["init_qpos"],
            ),
            seed,
            anchor,
            anchor_dir,
            freeze_episode=True,
        )
        if outcome.episode is None:
            raise RuntimeError(f"Frozen episode was not created for {pair_id}")
        episode = outcome.episode
        start_pose = episode["task"]["pickup_obj_start_pose"]
        goal_pose = episode["task"]["pickup_obj_goal_pose"]
        base_pose = episode["task"]["robot_base_pose"]
        init_qpos = episode["robot"]["init_qpos"]
        controlled = {
            "benchmark_id": BENCHMARK_ID,
            "split": split,
            "pair_id": pair_id,
            "kettle_pose_id": kettle["id"],
            "robot_pose_id": robot["id"],
            "kettle_pose_hash": pose_hash(start_pose),
            "robot_base_pose_hash": pose_hash(base_pose),
            "robot_state_hash": robot_state_hash(base_pose, init_qpos),
        }
        episode["controlled"] = controlled
        _assert_episode_contract(episode, anchor)
        episodes.append(episode)

        record = {
            "split": split,
            "episode_index": episode_index,
            "pair_id": pair_id,
            "seed": seed,
            "kettle_pose_id": kettle["id"],
            "robot_pose_id": robot["id"],
            "pickup_obj_start_pose": start_pose,
            "pickup_obj_goal_pose": goal_pose,
            "robot_base_pose": base_pose,
            "robot_init_qpos": init_qpos,
            "pose_hashes": {
                "pickup_start": pose_hash(start_pose),
                "pickup_goal": pose_hash(goal_pose),
                "robot_base": pose_hash(base_pose),
                "robot_init_qpos": stable_hash(init_qpos),
                "robot_state": robot_state_hash(base_pose, init_qpos),
                "pair": stable_hash(
                    {
                        "pickup_start": start_pose,
                        "pickup_goal": goal_pose,
                        "robot_base": base_pose,
                        "robot_init_qpos": init_qpos,
                    }
                ),
            },
            "supporting_geometry": outcome.support,
            "feasible_grasp_count": outcome.feasible_grasp_count,
            "robot_collision_free": True,
        }
        episode_manifest.append(record)
        generation_log.append(
            _attempt_log(
                "final_episode",
                seed,
                True,
                split=split,
                episode_index=episode_index,
                pair_id=pair_id,
                kettle_pose_id=kettle["id"],
                robot_pose_id=robot["id"],
                feasible_grasp_count=outcome.feasible_grasp_count,
            )
        )
    return episodes, episode_manifest


def _layout_manifest(
    split: str,
    layout: list[PairIndex],
    kettle_count: int,
    robot_count: int,
) -> dict[str, Any]:
    return {
        "split": split,
        "n_episodes": len(layout),
        "kettle_pose_count": kettle_count,
        "robot_pose_count": robot_count,
        "complete_cartesian": len(layout) == kettle_count * robot_count,
        "pairs": [
            {
                "pair_id": f"{split}_{pair.pair_id}",
                "kettle_index": pair.kettle_index,
                "robot_index": pair.robot_index,
            }
            for pair in layout
        ],
    }


def generate_benchmark(
    output_root: Path,
    anchor_dir: Path,
    master_seed: int,
    n_train: int,
    n_val: int,
    overwrite: bool,
) -> dict[str, Any]:
    active = active_v12_processes()
    if active:
        raise RuntimeError(
            "V12 trainers are active; generation is intentionally blocked to avoid contention. "
            "Run this command after V12 stops. First active command: "
            + active[0]
        )

    anchor = load_verified_anchor(anchor_dir)
    train_layout = build_pair_layout(
        n_train,
        DEFAULT_TRAIN_KETTLE_COUNT,
        DEFAULT_ROBOT_COUNT_PER_SPLIT,
    )
    val_layout = build_pair_layout(
        n_val,
        DEFAULT_VAL_KETTLE_COUNT,
        DEFAULT_ROBOT_COUNT_PER_SPLIT,
    )
    train_kettle_count, train_robot_count = required_axis_counts(train_layout)
    val_kettle_count, val_robot_count = required_axis_counts(val_layout)

    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output root already exists: {output_root}. Use --overwrite to replace it."
        )

    generation_log: list[dict[str, Any]] = []
    config = build_generation_config(anchor, master_seed, output_root)
    sampler = ControlledHouse0KettleSampler(config, anchor)
    try:
        train_kettles, val_kettles = _collect_kettle_candidates(
            sampler,
            anchor,
            anchor_dir,
            master_seed,
            train_kettle_count,
            val_kettle_count,
            generation_log,
        )
        all_kettles = train_kettles + val_kettles
        train_robots, val_robots = _collect_robot_candidates(
            sampler,
            anchor,
            anchor_dir,
            master_seed,
            train_robot_count,
            val_robot_count,
            all_kettles,
            generation_log,
        )
        train_episodes, train_episode_manifest = _build_split(
            "train",
            train_layout,
            train_kettles,
            train_robots,
            sampler,
            anchor,
            anchor_dir,
            master_seed,
            generation_log,
        )
        val_episodes, val_episode_manifest = _build_split(
            "val",
            val_layout,
            val_kettles,
            val_robots,
            sampler,
            anchor,
            anchor_dir,
            master_seed,
            generation_log,
        )
    finally:
        sampler.close()

    anchor_non_kettle = _non_kettle_scene(anchor.scene_modifications.object_poses)
    rejected = [record for record in generation_log if not record["accepted"]]
    manifest = {
        "schema_version": "1.0",
        "benchmark_id": BENCHMARK_ID,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "output_layout": {
            "train": "train/benchmark.json",
            "val": "val/benchmark.json",
            "manifest": "manifest.json",
            "generation_log": "generation_log.jsonl",
        },
        "source": {
            "benchmark_name": "FrankaPickDroidMiniBench",
            "benchmark_dir": str(anchor_dir),
            "benchmark_file_sha256": file_sha256(anchor_dir / "benchmark.json"),
            "episode_index": 0,
            "episode_role": "default episode 0 verified anchor/template",
            "episode": anchor.source.model_dump() if anchor.source is not None else None,
            "git_revisions": {
                "molmospaces": _source_git_revision(_MOLMOSPACES_ROOT),
                "rql": _source_git_revision(Path(__file__).resolve().parents[2]),
            },
        },
        "scene": {
            "dataset": anchor.scene_dataset,
            "data_split": anchor.data_split,
            "house_index": HOUSE_INDEX,
            "non_kettle_object_count": len(anchor_non_kettle),
            "non_kettle_scene_pose_hash": stable_hash(anchor_non_kettle),
        },
        "task": {
            "task_cls": TASK_CLASS,
            "task_type": "pick",
            "target_object": TARGET_OBJECT,
            "instruction": INSTRUCTION,
            "goal_definition": "pickup_obj_start_pose with exactly +0.05m Z",
            "goal_z_offset_m": GOAL_Z_OFFSET_M,
        },
        "seeds": {
            "master_seed": master_seed,
            "derivation": "sha256(master_seed:stage:indices), first 64 bits modulo 2^31-1",
            "episode_seeds": [
                record["seed"]
                for record in train_episode_manifest + val_episode_manifest
            ],
        },
        "requested_counts": {
            "n_train": n_train,
            "n_val": n_val,
            "default_n_train": DEFAULT_N_TRAIN,
            "default_n_val": DEFAULT_N_VAL,
        },
        "factorization": {
            "default_contract": {
                "train": "6 distinct kettle poses x 4 distinct robot states = 24",
                "val": (
                    "3 kettle poses disjoint from train x 4 robot states "
                    "disjoint from train = 12"
                ),
            },
            "smoke_semantics": (
                "--n-train/--n-val select a deterministic prefix of the corresponding "
                "default Cartesian grid; defaults require the complete grids"
            ),
            "train": _layout_manifest(
                "train",
                train_layout,
                train_kettle_count,
                train_robot_count,
            ),
            "val": _layout_manifest(
                "val",
                val_layout,
                val_kettle_count,
                val_robot_count,
            ),
        },
        "deduplication": {
            "rule": (
                "Every accepted pose must satisfy both the XY and wrapped-yaw minimum "
                "against every pose in both splits; the same checks enforce split leakage."
            ),
            "kettle_xy_threshold_m": KETTLE_XY_DEDUP_M,
            "kettle_yaw_threshold_rad": KETTLE_YAW_DEDUP_RAD,
            "kettle_yaw_threshold_deg": math.degrees(KETTLE_YAW_DEDUP_RAD),
            "robot_xy_threshold_m": ROBOT_XY_DEDUP_M,
            "robot_yaw_threshold_rad": ROBOT_YAW_DEDUP_RAD,
            "robot_yaw_threshold_deg": math.degrees(ROBOT_YAW_DEDUP_RAD),
            "pose_hash_decimals": POSE_HASH_DECIMALS,
        },
        "sampling_constraints": {
            "kettle_placement_radius_range_m": [
                KETTLE_PLACEMENT_MIN_RADIUS_M,
                KETTLE_PLACEMENT_MAX_RADIUS_M,
            ],
            "robot_sampling_radius_range_m": list(ROBOT_SAMPLE_RADIUS_RANGE_M),
            "robot_replay_max_distance_m": ROBOT_REPLAY_MAX_DISTANCE_M,
            "kettle_placement_attempts_per_call": KETTLE_PLACEMENT_ATTEMPTS,
            "max_candidate_attempts": MAX_CANDIDATE_ATTEMPTS,
        },
        "kettle_positions": {
            "train": train_kettles,
            "val": val_kettles,
        },
        "robot_positions": {
            "train": train_robots,
            "val": val_robots,
        },
        "episodes": train_episode_manifest + val_episode_manifest,
        "rejection_reasons": rejected,
        "generation_api_contract": [
            "PickTaskSampler",
            "place_object_near",
            "CPUMujocoEnv.place_robot_near",
            "get_pickup_grasps",
            "get_noncolliding_grasp_mask",
            "MlSpacesExpConfig.freeze_task_config",
            "frozen_config_to_episode_spec",
        ],
    }

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{output_root.name}.tmp-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "train").mkdir(parents=True)
    (staging / "val").mkdir(parents=True)
    try:
        (staging / "train" / "benchmark.json").write_text(
            json.dumps(train_episodes, indent=2) + "\n"
        )
        (staging / "val" / "benchmark.json").write_text(
            json.dumps(val_episodes, indent=2) + "\n"
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "generation_log.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in generation_log)
        )
        validate_output(staging, anchor_dir=anchor_dir)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(staging, output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return validate_output(output_root, anchor_dir=anchor_dir)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkValidationError(f"Failed to read {path}: {error}") from error


def _validate_pose_library(
    poses: list[list[float]],
    xy_threshold_m: float,
    yaw_threshold_rad: float,
    label: str,
) -> None:
    accepted: list[list[float]] = []
    for index, pose in enumerate(poses):
        distinct, reason = pose_is_distinct(
            pose,
            accepted,
            xy_threshold_m,
            yaw_threshold_rad,
        )
        if not distinct:
            raise BenchmarkValidationError(f"{label}[{index}] violates dedup: {reason}")
        accepted.append(pose)


def _validate_factorization(
    manifest: dict[str, Any],
    split: str,
    episodes: list[dict[str, Any]],
) -> None:
    split_factorization = manifest["factorization"][split]
    pairs = split_factorization["pairs"]
    if len(pairs) != len(episodes):
        raise BenchmarkValidationError(
            f"{split} factorization has {len(pairs)} pairs for {len(episodes)} episodes"
        )
    expected_ids = [pair["pair_id"] for pair in pairs]
    actual_ids = [episode.get("controlled", {}).get("pair_id") for episode in episodes]
    if actual_ids != expected_ids:
        raise BenchmarkValidationError(
            f"{split} pair order mismatch: expected {expected_ids}, got {actual_ids}"
        )

    requested = manifest["requested_counts"][f"n_{split}"]
    if split == "train" and requested == DEFAULT_N_TRAIN:
        expected = {
            (kettle_index, robot_index)
            for kettle_index, robot_index in itertools.product(
                range(DEFAULT_TRAIN_KETTLE_COUNT),
                range(DEFAULT_ROBOT_COUNT_PER_SPLIT),
            )
        }
        actual = {
            (pair["kettle_index"], pair["robot_index"]) for pair in pairs
        }
        if actual != expected:
            raise BenchmarkValidationError("Default train output is not the required 6x4 grid")
    if split == "val" and requested == DEFAULT_N_VAL:
        expected = {
            (kettle_index, robot_index)
            for kettle_index, robot_index in itertools.product(
                range(DEFAULT_VAL_KETTLE_COUNT),
                range(DEFAULT_ROBOT_COUNT_PER_SPLIT),
            )
        }
        actual = {
            (pair["kettle_index"], pair["robot_index"]) for pair in pairs
        }
        if actual != expected:
            raise BenchmarkValidationError("Default val output is not the required 3x4 grid")


def validate_output(
    output_root: Path,
    anchor_dir: Path | None = None,
) -> dict[str, Any]:
    """Schema-validate existing files and check all controlled invariants."""

    output_root = output_root.resolve()
    required_files = (
        output_root / "train" / "benchmark.json",
        output_root / "val" / "benchmark.json",
        output_root / "manifest.json",
        output_root / "generation_log.jsonl",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise BenchmarkValidationError("Missing required output files: " + ", ".join(missing))

    manifest = _load_json(output_root / "manifest.json")
    required_manifest_keys = {
        "schema_version",
        "benchmark_id",
        "source",
        "scene",
        "task",
        "seeds",
        "requested_counts",
        "factorization",
        "deduplication",
        "kettle_positions",
        "robot_positions",
        "episodes",
        "rejection_reasons",
    }
    missing_manifest_keys = sorted(required_manifest_keys - set(manifest))
    if missing_manifest_keys:
        raise BenchmarkValidationError(
            f"Manifest missing required keys: {missing_manifest_keys}"
        )
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise BenchmarkValidationError(
            f"manifest benchmark_id={manifest.get('benchmark_id')!r}, expected {BENCHMARK_ID!r}"
        )
    expected_manifest_task = {
        "task_cls": TASK_CLASS,
        "task_type": "pick",
        "target_object": TARGET_OBJECT,
        "instruction": INSTRUCTION,
        "goal_z_offset_m": GOAL_Z_OFFSET_M,
    }
    if any(
        manifest["task"].get(key) != value
        for key, value in expected_manifest_task.items()
    ):
        raise BenchmarkValidationError("Manifest task contract is incorrect")
    if (
        manifest["scene"].get("dataset") != SCENE_DATASET
        or manifest["scene"].get("house_index") != HOUSE_INDEX
    ):
        raise BenchmarkValidationError("Manifest scene contract is incorrect")
    expected_dedup = {
        "kettle_xy_threshold_m": KETTLE_XY_DEDUP_M,
        "kettle_yaw_threshold_rad": KETTLE_YAW_DEDUP_RAD,
        "robot_xy_threshold_m": ROBOT_XY_DEDUP_M,
        "robot_yaw_threshold_rad": ROBOT_YAW_DEDUP_RAD,
    }
    if any(
        not math.isclose(
            float(manifest["deduplication"].get(key, math.nan)),
            value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for key, value in expected_dedup.items()
    ):
        raise BenchmarkValidationError("Manifest deduplication thresholds are incorrect")
    if anchor_dir is None:
        source_path = manifest.get("source", {}).get("benchmark_dir")
        anchor_dir = find_anchor_benchmark(Path(source_path) if source_path else None)
    anchor = load_verified_anchor(anchor_dir)
    if manifest.get("source", {}).get("episode_index") != 0:
        raise BenchmarkValidationError("Manifest source episode_index must be 0")
    anchor_file_hash = file_sha256(anchor_dir / "benchmark.json")
    if manifest.get("source", {}).get("benchmark_file_sha256") != anchor_file_hash:
        raise BenchmarkValidationError("Manifest source benchmark hash differs from anchor")

    train_raw = _load_json(output_root / "train" / "benchmark.json")
    val_raw = _load_json(output_root / "val" / "benchmark.json")
    if not isinstance(train_raw, list) or not isinstance(val_raw, list):
        raise BenchmarkValidationError("Each benchmark.json must contain a JSON list")

    train_loaded = load_all_episodes(output_root / "train")
    val_loaded = load_all_episodes(output_root / "val")
    if len(train_loaded) != len(train_raw) or len(val_loaded) != len(val_raw):
        raise BenchmarkValidationError("load_all_episodes count differs from raw JSON count")
    if len(train_raw) != manifest["requested_counts"]["n_train"]:
        raise BenchmarkValidationError("Train episode count differs from manifest")
    if len(val_raw) != manifest["requested_counts"]["n_val"]:
        raise BenchmarkValidationError("Val episode count differs from manifest")

    _validate_factorization(manifest, "train", train_raw)
    _validate_factorization(manifest, "val", val_raw)

    all_raw = [("train", index, episode) for index, episode in enumerate(train_raw)]
    all_raw.extend(("val", index, episode) for index, episode in enumerate(val_raw))
    episode_manifest = {
        (record["split"], record["episode_index"]): record
        for record in manifest["episodes"]
    }
    if len(episode_manifest) != len(all_raw):
        raise BenchmarkValidationError("Manifest episode records are missing or duplicated")
    train_kettles = manifest["kettle_positions"]["train"]
    val_kettles = manifest["kettle_positions"]["val"]
    train_robots = manifest["robot_positions"]["train"]
    val_robots = manifest["robot_positions"]["val"]
    kettle_by_id = {
        candidate["id"]: candidate for candidate in train_kettles + val_kettles
    }
    robot_by_id = {
        candidate["id"]: candidate for candidate in train_robots + val_robots
    }

    seeds = []
    pair_hashes = set()
    for split, index, episode in all_raw:
        _assert_episode_contract(episode, anchor)
        controlled = episode.get("controlled")
        if not isinstance(controlled, dict) or controlled.get("split") != split:
            raise BenchmarkValidationError(f"{split}[{index}] lacks controlled split metadata")
        record = episode_manifest.get((split, index))
        if record is None:
            raise BenchmarkValidationError(f"No manifest record for {split}[{index}]")
        if controlled["pair_id"] != record["pair_id"]:
            raise BenchmarkValidationError(f"Pair ID mismatch for {split}[{index}]")
        if episode["seed"] != record["seed"]:
            raise BenchmarkValidationError(f"Seed mismatch for {split}[{index}]")
        seeds.append(episode["seed"])

        start = episode["task"]["pickup_obj_start_pose"]
        goal = episode["task"]["pickup_obj_goal_pose"]
        base = episode["task"]["robot_base_pose"]
        init_qpos = episode["robot"]["init_qpos"]
        kettle_candidate = kettle_by_id.get(controlled.get("kettle_pose_id"))
        robot_candidate = robot_by_id.get(controlled.get("robot_pose_id"))
        if kettle_candidate is None or kettle_candidate["split"] != split:
            raise BenchmarkValidationError(
                f"Unknown or cross-split kettle ID for {split}[{index}]"
            )
        if robot_candidate is None or robot_candidate["split"] != split:
            raise BenchmarkValidationError(
                f"Unknown or cross-split robot ID for {split}[{index}]"
            )
        if not np.allclose(
            start,
            kettle_candidate["pose"],
            atol=POSE_ATOL,
            rtol=0.0,
        ):
            raise BenchmarkValidationError(
                f"Kettle factor replay mismatch for {split}[{index}]"
            )
        if not np.allclose(
            base,
            robot_candidate["base_pose"],
            atol=POSE_ATOL,
            rtol=0.0,
        ) or stable_hash(init_qpos) != stable_hash(robot_candidate["init_qpos"]):
            raise BenchmarkValidationError(
                f"Robot factor replay mismatch for {split}[{index}]"
            )
        expected_hashes = {
            "pickup_start": pose_hash(start),
            "pickup_goal": pose_hash(goal),
            "robot_base": pose_hash(base),
            "robot_init_qpos": stable_hash(init_qpos),
            "robot_state": robot_state_hash(base, init_qpos),
            "pair": stable_hash(
                {
                    "pickup_start": start,
                    "pickup_goal": goal,
                    "robot_base": base,
                    "robot_init_qpos": init_qpos,
                }
            ),
        }
        if expected_hashes != record["pose_hashes"]:
            raise BenchmarkValidationError(f"Pose hash mismatch for {split}[{index}]")
        expected_controlled_hashes = {
            "kettle_pose_hash": expected_hashes["pickup_start"],
            "robot_base_pose_hash": expected_hashes["robot_base"],
            "robot_state_hash": expected_hashes["robot_state"],
        }
        if any(
            controlled.get(key) != value
            for key, value in expected_controlled_hashes.items()
        ):
            raise BenchmarkValidationError(
                f"Controlled pose hash mismatch for {split}[{index}]"
            )
        if expected_hashes["pair"] in pair_hashes:
            raise BenchmarkValidationError(f"Duplicate full pair at {split}[{index}]")
        pair_hashes.add(expected_hashes["pair"])
        if int(record["feasible_grasp_count"]) <= 0 or not record["robot_collision_free"]:
            raise BenchmarkValidationError(f"Invalid simulation checks for {split}[{index}]")

    if len(seeds) != len(set(seeds)):
        raise BenchmarkValidationError("Episode seeds are not unique")
    if seeds != manifest["seeds"]["episode_seeds"]:
        raise BenchmarkValidationError("Manifest episode seed list differs from episodes")
    if not isinstance(manifest["seeds"].get("master_seed"), int):
        raise BenchmarkValidationError("Manifest master_seed must be an integer")

    all_kettle_poses = [
        candidate["pose"] for candidate in train_kettles + val_kettles
    ]
    if len(train_kettles) != manifest["factorization"]["train"]["kettle_pose_count"]:
        raise BenchmarkValidationError("Train kettle candidate count differs from factorization")
    if len(val_kettles) != manifest["factorization"]["val"]["kettle_pose_count"]:
        raise BenchmarkValidationError("Val kettle candidate count differs from factorization")
    for candidate in train_kettles + val_kettles:
        if candidate["pose_hash"] != pose_hash(candidate["pose"]):
            raise BenchmarkValidationError(
                f"Kettle candidate pose hash mismatch: {candidate['id']}"
            )
        support = candidate.get("supporting_geometry", {})
        if support.get("available") and "geom_id" not in support:
            raise BenchmarkValidationError(
                f"Incomplete supporting geometry metadata: {candidate['id']}"
            )
    _validate_pose_library(
        all_kettle_poses,
        KETTLE_XY_DEDUP_M,
        KETTLE_YAW_DEDUP_RAD,
        "kettle_positions",
    )
    if len({candidate["pose_hash"] for candidate in train_kettles}) != len(train_kettles):
        raise BenchmarkValidationError("Duplicate train kettle pose hashes")
    if {candidate["pose_hash"] for candidate in train_kettles} & {
        candidate["pose_hash"] for candidate in val_kettles
    }:
        raise BenchmarkValidationError("Kettle pose hash leakage between train and val")

    all_robot_poses = [
        candidate["base_pose"] for candidate in train_robots + val_robots
    ]
    if len(train_robots) != manifest["factorization"]["train"]["robot_pose_count"]:
        raise BenchmarkValidationError("Train robot candidate count differs from factorization")
    if len(val_robots) != manifest["factorization"]["val"]["robot_pose_count"]:
        raise BenchmarkValidationError("Val robot candidate count differs from factorization")
    for candidate in train_robots + val_robots:
        if candidate["base_pose_hash"] != pose_hash(candidate["base_pose"]):
            raise BenchmarkValidationError(
                f"Robot candidate base hash mismatch: {candidate['id']}"
            )
        if candidate["init_qpos_hash"] != stable_hash(candidate["init_qpos"]):
            raise BenchmarkValidationError(
                f"Robot candidate qpos hash mismatch: {candidate['id']}"
            )
        if candidate["robot_state_hash"] != robot_state_hash(
            candidate["base_pose"],
            candidate["init_qpos"],
        ):
            raise BenchmarkValidationError(
                f"Robot candidate state hash mismatch: {candidate['id']}"
            )
    _validate_pose_library(
        all_robot_poses,
        ROBOT_XY_DEDUP_M,
        ROBOT_YAW_DEDUP_RAD,
        "robot_positions",
    )
    train_robot_hashes = {candidate["robot_state_hash"] for candidate in train_robots}
    val_robot_hashes = {candidate["robot_state_hash"] for candidate in val_robots}
    if len(train_robot_hashes) != len(train_robots):
        raise BenchmarkValidationError("Duplicate train robot state hashes")
    if len(val_robot_hashes) != len(val_robots):
        raise BenchmarkValidationError("Duplicate val robot state hashes")
    if train_robot_hashes & val_robot_hashes:
        raise BenchmarkValidationError("Robot state leakage between train and val")

    non_kettle_hash = stable_hash(
        _non_kettle_scene(anchor.scene_modifications.object_poses)
    )
    if non_kettle_hash != manifest["scene"]["non_kettle_scene_pose_hash"]:
        raise BenchmarkValidationError("Manifest non-kettle scene hash differs from anchor")

    log_records = []
    for line_number, line in enumerate(
        (output_root / "generation_log.jsonl").read_text().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise BenchmarkValidationError(
                f"Invalid generation_log.jsonl line {line_number}: {error}"
            ) from error
        if not {"stage", "seed", "accepted"}.issubset(record):
            raise BenchmarkValidationError(
                f"generation_log.jsonl line {line_number} lacks required fields"
            )
        if not record["accepted"] and not record.get("reason"):
            raise BenchmarkValidationError(
                f"Rejected generation log line {line_number} lacks a reason"
            )
        log_records.append(record)
    rejected = [record for record in log_records if not record.get("accepted", False)]
    if rejected != manifest["rejection_reasons"]:
        raise BenchmarkValidationError(
            "Manifest rejection_reasons differs from generation_log.jsonl"
        )

    return {
        "benchmark_id": BENCHMARK_ID,
        "output_root": str(output_root),
        "train_episodes": len(train_raw),
        "val_episodes": len(val_raw),
        "load_all_episodes_compatible": True,
        "non_kettle_scene_pose_hash": non_kettle_hash,
        "rejection_count": len(rejected),
        "validated": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--anchor-benchmark", type=Path, default=None)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument(
        "--n-train",
        type=int,
        default=DEFAULT_N_TRAIN,
        help="Train episodes; defaults enforce the complete 6x4 factorization",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=DEFAULT_N_VAL,
        help="Validation episodes; defaults enforce the complete disjoint 3x4 factorization",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing JSON outputs without creating a simulator",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output root after successful staged validation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchor_dir = find_anchor_benchmark(args.anchor_benchmark)
    if args.validate_only:
        summary = validate_output(args.output_root, anchor_dir=anchor_dir)
    else:
        summary = generate_benchmark(
            output_root=args.output_root.resolve(),
            anchor_dir=anchor_dir,
            master_seed=args.master_seed,
            n_train=args.n_train,
            n_val=args.n_val,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
