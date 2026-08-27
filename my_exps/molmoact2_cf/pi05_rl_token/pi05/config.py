"""Every knob for the pi0.5 / RL-Token experiments, in one place.

Nothing that changes a result is hidden in a command line. Edit a value here, run an
entry point, and the run uses it:

    scripts/run_eval.py     evaluate the frozen VLA, or an RL actor on top of it
    scripts/run_train.py    train RL Token on a scene

Both entry points take at most a scene name and a few machine-level overrides (which GPU,
which port). Everything that decides what the numbers mean -- chunk size, the execution
regime, the gate, the RL hyperparameters -- is a field of `EvalConfig` or `RLConfig`
below.

Each number that came from a measurement says so. Anything marked "provisional" is a
default nobody has tested, kept explicit so it is never mistaken for a finding.

    from pi05.config import EvalConfig, RLConfig, SCENES, describe
    print(describe())
    cfg = EvalConfig(scene="house21", chunk_size=1, episodes=36)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# ======================================================================================
# Paths. Everything this project produced lives under one root.
# ======================================================================================

ROOT = Path("/home/jovyan/users/staroverov/pi05_molmospaces")
#: Copy of method_cards/rl_token, used so this tree can be edited without touching
#: the original. Checkpoint / venvs still live under ROOT (14 GB, not duplicated).
CODE = Path(
    "/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/"
    "molmoact2_cf/pi05_rl_token"
)

#: pi0.5 fine-tuned on MolmoSpaces Pick, converted to PyTorch. This is the frozen VLA.
CHECKPOINT = ROOT / "checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999"

#: openpi training-config name the checkpoint was produced under.
OPENPI_CONFIG = "pi05_droid_finetune"

#: Phase-1 RL-token autoencoders for the selected scenes, trained on the corpus collected
#: in the chunk 8 + step_time regime: ae_atomizer, ae_sink_dispenser, ae_desk_mug and
#: ae_combined.
#:
#: The previous directory, pi05_runs/token_ae, holds encoders fit to a corpus collected
#: under chunk 1 + plan_time -- a policy that scored 40.6% and visited different states.
#: Its per-scene names differ (ae_house2 and so on) so a stale one fails loudly, but
#: `ae_combined` exists in BOTH, and that one would have been picked up in silence.
#: Copied ae_desk_mug.pt lives here. Other scene encoders are still under ROOT if needed.
TOKEN_AE_DIR = CODE / "assets"

# Three interpreters, not interchangeable.
#: Serving pi0.5. openpi's own venv is NOT patched and the model refuses to load in it.
VENV_TORCH = ROOT / "venvs/pi05_torch/bin/python"
#: The simulator and the `rlt` package.
VENV_SIM = Path("/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python")
#: JAX side: dataset tooling and the original eval driver.
VENV_OPENPI = Path("/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/openpi/.venv/bin/python")

MOLMOSPACES = Path("/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces")
MLSPACES_ASSETS = Path("/home/jovyan/users/staroverov/B1K/mlspaces/assets")
MLSPACES_CACHE = Path("/home/jovyan/users/staroverov/B1K/mlspaces/cache")

#: The shipped 1000-episode validation benchmark. Episodes 0-127 are the held-out slice
#: every published number in this project was measured on.
VAL_BENCHMARK = (
    MLSPACES_CACHE / "benchmarks/molmospaces-bench-v1/20260408/procthor-10k"
    / "FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
)

#: The HF cache must live here. Under /home/user it fills the container disk.
HF_HOME = Path("/home/jovyan/users/staroverov/.cache/huggingface")

#: Scratch for simulator episode output, deleted per episode.
TMP_ROLLOUT_DIR = Path("/home/jovyan/users/staroverov/B1K/tmp/pi05")


# ======================================================================================
# Fixed by the model, the data, or the environment. Not tunable.
# ======================================================================================

#: Actions the model emits per call (openpi pi05_droid_finetune action_horizon).
VLA_CHUNK = 16

#: Dimensions the caller sees: 7 arm joints plus gripper.
ACTION_DIM = 8
ARM_DOF = 7

#: Width of the frozen VLA token sequence. The pi0.5 PaliGemma backbone is gemma_2b, so
#: 2048 -- not the 2560 of MolmoAct2. Phase-1 encoders are built for a fixed width and
#: the two cannot be swapped. Every pi0.5 RL run must export RLT_VLA_TOKEN_DIM=2048.
VLA_TOKEN_DIM = 2048

#: The RL state also carries proprioception: 7 arm positions + gripper + their velocities.
PROPRIO_DIM = 16

#: The prefix token sequence pi0.5 produces per step, measured: (968, 2048).
TOKEN_SEQ_LEN = 968


# ======================================================================================
# Scenes. Each is one benchmark episode repeated with a +-2 cm horizontal jitter on the
# object, built by scripts/make_repeat_benchmark.py.
# ======================================================================================


@dataclass(frozen=True)
class Scene:
    """One task the experiments run on.

    Each scene has two disjoint sets of jittered repeats, built by
    scripts/make_repeat_benchmark.py from the same val episode with different seeds:

        benchmark_train   48 repeats, seed 0     RL training, and token collection
        benchmark_eval    48 repeats, seed 1000  every number that gets reported

    They share no object pose (the generator keeps the unjittered template as repeat 0,
    so the eval set was built with 49 and that one dropped). The split exists because the
    original per-scene benchmarks held 12 repeats for house2 and house10 and the RL pool
    was 0-11 -- i.e. every episode, leaving nothing held out to measure an actor on.
    """

    name: str
    val_episode: int  #: index into the shipped val benchmark
    house: int
    task: str
    benchmark_train: Path
    benchmark_eval: Path
    note: str = ""
    #: Catalog handover used when a run sets gate_step=-1. 0 means none measured.
    gate_step: int = 0
    #: The pre-split directory (12 repeats for house2/house10, 40 for house21). Kept only
    #: to reproduce measurements taken before 2026-08-20; it overlaps benchmark_train.
    legacy_benchmark: Path | None = None


SCENES: dict[str, Scene] = {
    "house2": Scene(
        name="house2",
        val_episode=117,
        house=2,
        task="pick up the bottle.",
        benchmark_train=ROOT / "bench_house2_train48",
        benchmark_eval=ROOT / "bench_house2_eval48",
        legacy_benchmark=ROOT / "bench_collect_house2",
        note="soap dispenser on a toilet lid, elongated. 0/76 successes at chunk 1 + "
             "plan_time: the policy reaches the right pose and pushes instead of "
             "grasping. Kept deliberately -- it tests whether RL can start from zero "
             "reward. RE-MEASURE before relying on that: the 0/76 predates the chunk 8 + "
             "step_time regime, and if the scene now scores above zero it is no longer "
             "the zero-reward test it was chosen to be.",
    ),
    "house10": Scene(
        name="house10",
        val_episode=2,
        house=10,
        task="pick up the ladle.",
        benchmark_train=ROOT / "bench_house10_train48",
        benchmark_eval=ROOT / "bench_house10_eval48",
        legacy_benchmark=ROOT / "bench_collect_house10",
        note="36.1% at chunk 1, 83.3% at chunk 8 -- both under plan_time; re-measure "
             "under the chunk 8 + step_time default before comparing.",
    ),
    "house21": Scene(
        name="house21",
        val_episode=127,
        house=21,
        task="pick up the pot.",
        benchmark_train=ROOT / "bench_house21_train48",
        benchmark_eval=ROOT / "bench_house21_eval48",
        legacy_benchmark=ROOT / "bench_collect_house21",
        note="the opposite trend under plan_time: 88.9% at chunk 1, 58.3% at 8. "
             "Re-measure under the chunk 8 + step_time default before comparing.",
    ),
    # --- selected 2026-08-20 by measurement in the chunk 8 + step_time regime -----------
    "wine_bottle": Scene(
        name="wine_bottle",
        val_episode=41,
        house=126,
        task="pick up the bottle.",
        benchmark_train=ROOT / "bench_wine_bottle_train48",
        benchmark_eval=ROOT / "bench_wine_bottle_eval48",
        gate_step=48,  # the gripper is on the cap by step 45-60 and the stall sets in around 90, so 48 hands RL the whole grasp phase
        note="REJECTED 2026-08-20: WRONG OBJECT. It scores 0/64, but the recorded target "
             "image points show the wine bottle sitting in the corner of the wrist view "
             "while the fingers close on a large gold object nearby -- and the hand ends "
             "at the target on only 9 of 64 rollouts. A correction applied to the action "
             "chunk cannot change which object the model chose. Superseded by 'atomizer'. "
             "This was first accepted on an eyeballed frame, which is why "
             "scripts/check_target.py exists.",
    ),
    "sink_dispenser": Scene(
        name="sink_dispenser",
        val_episode=25,
        house=112,
        task="pick up the bottle.",
        benchmark_train=ROOT / "bench_sink_dispenser_train48",
        benchmark_eval=ROOT / "bench_sink_dispenser_eval48",
        gate_step=40,  # the arm drifts off the dispenser right after its first approach at ~45, so RL has to be in control before the drift
        note="MIDDLE band: probed 3/12 = 25%. Reaches the dispenser by the sink, then "
             "stalls from about step 80 with the fingers on the body instead of the cap.",
    ),
    "atomizer": Scene(
        name="atomizer",
        val_episode=14,
        house=106,
        task="pick up the bottle.",
        benchmark_train=ROOT / "bench_atomizer_train48",
        benchmark_eval=ROOT / "bench_atomizer_eval48",
        gate_step=56,  # the atomizer enters the wrist view at ~30 and the hand is on
                       # it by ~60; both failure modes diverge over 60-75 -- one stalls
                       # on the object, the other drifts off it -- so 56 is the last
                       # chunk boundary before either can happen
        note="ZERO band: probed 0/12, and 6 of those 12 end with the hand ON the object "
             "(target visible 100%, gap 0.35-0.55 -- inside the range every measured "
             "success falls in) and it still never lifts. Confirmed by drawing the "
             "recorded target image points on the wrist frame: the atomizer sits between "
             "the fingers at step 400, ungrasped. A grasp failure, not a wrong-object one.",
    ),
    # --- multi-task: 20 different episodes, not one repeated ------------------------------
    "multi20": Scene(
        name="multi20",
        val_episode=-1,  # not one episode; see bench_multi20/selection.json for the 20
        house=-1,        # 20 distinct houses
        task="pick up the object.",
        benchmark_train=ROOT / "bench_multi20",
        benchmark_eval=ROOT / "bench_multi20",
        gate_step=40,
        note="MULTI-TASK, probed only as its parts: 20 val episodes drawn at random with "
             "seed 20260821 from the first 128, spanning 20 houses and 10 object types. "
             "Coarse bands of the draw: 15 good, 2 low, 2 zero, 1 unmeasured -- half sit "
             "near 100%, so the aggregate mostly asks whether one actor across many tasks "
             "keeps what already works. Report it per episode. Gate 40 is the earliest of "
             "the three gates measured from video (40/56/56): with no per-scene footage "
             "here, handing over early only gives RL more of the approach, while handing "
             "over late would leave it unable to touch the failure at all.",
    ),
    "desk_mug": Scene(
        name="desk_mug",
        val_episode=9,
        house=104,
        task="pick up the mug.",
        benchmark_train=CODE / "assets/benches/bench_desk_mug_train48",
        benchmark_eval=CODE / "assets/benches/bench_desk_mug_eval48",
        gate_step=56,  # the mug enters the wrist view at 60 and the successful and failing rollouts diverge over 60-75, so 56 is the last boundary before that
        note="UPPER band: probed 6/12 = 50%. Reaches the mug on the desk plate; successes "
             "close by about step 95, failures park beside it and run the full horizon -- "
             "a grasp failure on the right object, so there is something for RL to fix.",
    ),
    "kettle": Scene(
        name="kettle",
        val_episode=0,
        house=0,
        task="pick up the kettle.",
        benchmark_train=CODE / "assets/benches/house0_kettle_v21_pose0/train",
        benchmark_eval=CODE / "assets/benches/house0_kettle_v21_pose0/eval",
        gate_step=40,  # PI05 default; kettle contact sheet not measured
        note="house0 pick up the kettle, HARD pose train_k00 with ±2 cm XY jitter "
             "(same construction as desk_mug). Train seed 0 / eval seed 1000, 48 specs. "
             "Horizon 400. Gate 40.",
    ),
}


#: The scenes the RL matrix runs on. Selected by measurement; `SCENES` also still holds
#: the three inherited ones, whose selection came from a profiling run recorded as
#: contaminated and whose rates were measured under a regime the project no longer runs.
#: Naming the selection explicitly is what stops those being picked up by default.
SELECTED_SCENES: list[str] = ["atomizer", "sink_dispenser", "desk_mug"]


def _pick_object_scenes() -> dict[str, Scene]:
    """Mug/kettle-style benches for the other Pick-v1.1 object categories.

    Built by scripts/make_pick_object_benches.py. Mug and kettle stay as desk_mug /
    kettle; these 16 are the remaining metadata categories, one held-out episode each.
    """
    manifest_path = CODE / "assets/benches/pick_objects/manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text())
    scenes: dict[str, Scene] = {}
    for item in payload["objects"]:
        name = item["name"]
        root = CODE / "assets/benches/pick_objects" / name
        scenes[name] = Scene(
            name=name,
            val_episode=int(item["val_episode"]),
            house=int(item["house"]),
            task=item["task"],
            benchmark_train=root / "train",
            benchmark_eval=root / "eval",
            gate_step=int(item.get("gate_step", 0)),
            note=item["note"],
        )
    return scenes


SCENES.update(_pick_object_scenes())



# ======================================================================================
# How a predicted chunk is executed.
#
# SETTLED 2026-08-20 by a sweep over the held-out 128-episode val benchmark: one server
# per cell, eight cells in parallel, nothing varied but these two settings.
#
#                chunk1    chunk4    chunk8   chunk16
#     plan_time    40.6%     51.6%     44.5%      0.0%
#     step_time    37.5%     51.6%     62.5%     63.3%
#
# Read it by row, not by column: what decides the outcome is `conversion`, not chunk
# length. Under the faithful regime longer chunks help and then saturate; under plan_time
# the accumulated lag eventually destroys the policy outright.
#
# Fisher exact, two-sided:
#     chunk1/plan_time  vs chunk16/step_time   40.6% -> 63.3%   p=0.0004
#     chunk1/plan_time  vs chunk8/step_time    40.6% -> 62.5%   p=0.0007
#     chunk8/plan_time  vs chunk8/step_time    44.5% -> 62.5%   p=0.0057
#     chunk16/plan_time vs chunk16/step_time    0.0% -> 63.3%   p=4e-33
#     chunk8/step_time  vs chunk16/step_time   62.5% vs 63.3%   p=1.0  (tied)
#
# Two calibration notes for reading any future number against this table:
#
#   * chunk1/plan_time and chunk1/step_time are the SAME configuration -- the two
#     conversions coincide at chunk 1, which test_pi05_model.py proves. They ran as
#     separate jobs, so their gap is pure run-to-run noise from the flow-matching
#     sampler: 40.6% vs 37.5%, p=0.70. Treat gaps below ~5 points as noise.
#   * chunk1/plan_time reproduced the previously published PyTorch-converted baseline of
#     40.6% exactly. That is the evidence that moving the conversion out of the server
#     did not change behaviour.
#
# The earlier per-scene table (house10 36->83%, house21 89->58% across chunk 1..8) was
# measured entirely under plan_time. It describes how the lag interacts with two
# particular scenes, not how chunk length behaves; do not read a policy out of it.
# ======================================================================================

#: Actions of each predicted chunk executed before re-planning.
#:
#: 8, measured. Statistically tied with 16 (p=1.0) and chosen over it because it re-plans
#: twice as often for the same success rate and the same wall clock, so it reacts sooner
#: when the scene does something the plan did not expect. It is also ~40% faster end to
#: end than chunk 1 (5519 s vs 9367 s for 128 episodes): most of the cost is the VLA
#: forward pass, and a longer chunk needs fewer of them.
#:
#: A long chunk is not a compromise here, and 1 was never "the formally correct value" --
#: that framing conflated two things. Each action is labelled as a displacement from its
#: own step's state, i.e. a joint velocity, and a velocity chunk is meant to run open
#: loop. What must hold is that action[k] be resolved against the state at step t+k,
#: which is `conversion`, below -- not chunk length.
DEFAULT_CHUNK_SIZE = 8

#: When the predicted joint deltas become the absolute targets the environment executes.
#: This is the knob that decides whether a long chunk is faithful to the labels.
#:
#: "step_time"  each action is added to the arm state observed at the step it executes:
#:              command[t+k] = state[t+k] + action[k], which reconstructs the commanded
#:              target exactly for every k. Faithful at any chunk length, and now the
#:              measured winner: +18 points over plan_time at chunk 8 (p=0.0057).
#: "plan_time"  every action is added to the arm state observed when the chunk was
#:              planned. Exact at k=0 only; after that it is short by the distance
#:              travelled since the plan -- 0.027 rad per step, reaching 0.32 rad by
#:              k=15, larger than the motion being commanded. At chunk 16 it scores
#:              0/128: the arm is driven somewhere the plan never intended.
#:
#: The two are identical at chunk_size=1 and diverge as the chunk grows. plan_time is
#: kept only to reproduce pre-refactor runs: every number measured before 2026-08-20,
#: including the per-scene chunk table, was produced under it.
DEFAULT_CONVERSION = "step_time"

#: Demonstrations command the gripper fully open or fully closed and nothing between, so
#: a threshold reproduces them exactly.
DEFAULT_GRASPING = "binary"
DEFAULT_GRASP_THRESHOLD = 0.5


# ======================================================================================
# RL Token
# ======================================================================================

#: Step at which RL takes control of the arm, chosen by the project owner from watching
#: rollouts. Latched: once open it stays open for the rest of the episode.
#:
#: A distance gate was tried first and abandoned for a measured reason: the evaluator
#: records the object pose from the benchmark spec, not its live position, so the number
#: stops describing reality as soon as the policy nudges the object. On an elongated
#: object the centre also stays far while the gripper is at a graspable end. Do not
#: resurrect the spec-pose version; a real distance gate needs the live MuJoCo pose
#: pulled out during the rollout.
GATE_STEP = 40
GATE_LATCH = True
#: Fraction of the episode the frozen VLA runs before RL takes over. 0 keeps
#: RL from step 0. 0.1 is the PI05 default (~10% of horizon), snapped down to a
#: chunk boundary so the handover is a decision point: 0.1×500 → 48, 0.1×400 → 40.
GATE_FRAC = 0.0


def gate_from_frac(frac: float, horizon: int, chunk_size: int) -> int:
    """VLA prefix as a fraction of the episode, snapped down to a chunk boundary.

    10% of 500 is 50, which is mid-chunk; snapping down (48) starts the actor no
    later than the requested fraction. 0 disables the prefix.
    """
    if frac <= 0:
        return 0
    if frac >= 1:
        raise ValueError(f"gate_frac must be in [0, 1), got {frac}")
    return int(horizon * frac) // int(chunk_size) * int(chunk_size)


def _check_gate_frac(frac: float) -> str:
    """Empty when valid, else the reason. NaN fails the range check."""
    if not 0.0 <= float(frac) < 1.0:
        return f"gate_frac must be in [0, 1), got {frac}"
    return ""


def _resolve_gate_step(
    explicit: int,
    scene: Scene | None,
    *,
    gate_frac: float = 0.0,
    horizon: int = 500,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """0 = RL from the first env step. N = VLA prefix of N steps. -1 = scene catalog.

    `gate_frac > 0` wins over `gate_step`: the prefix is a fraction of `horizon`.
    """
    if gate_frac:
        return gate_from_frac(gate_frac, horizon, chunk_size)
    if explicit >= 0:
        return explicit
    if explicit != -1:
        raise ValueError(f"gate_step must be >= -1, got {explicit}")
    catalog = 0 if scene is None else int(scene.gate_step)
    if not catalog:
        raise ValueError(
            "gate_step=-1 needs a scene catalog gate; this scene has none. "
            "Set gate_step=0 for RL from step 0, gate_frac=0.1 for a 10% VLA prefix, "
            "or N for a VLA prefix of N steps."
        )
    return catalog

#: Which space the RL actor refines.
#:
#: "absolute"  the reference chunk is absolute joint targets (the server converts before
#:             the actor sees it). This is what the discarded 8-run matrix used.
#: "delta"     the reference chunk is the raw joint deltas the model emits; conversion
#:             happens after the actor, at the environment boundary.
#:
#: This is not cosmetic. The actor is a plain MLP that emits the chunk outright (see
#: rlt/networks.py) rather than a residual, so at initialisation it outputs values near
#: zero. Near-zero deltas are a near-no-op; near-zero absolute joint targets are a
#: completely different arm pose. The exploration std and the beta pull toward the
#: reference are likewise scale-sensitive: sigma=0.02 is large next to deltas of ~0.027
#: rad and small next to absolute targets of ~2 rad.
#:
#: "delta", chosen by the project owner on 2026-08-20, with the requirement that the
#: actor be shown exactly the actions that would be executed at real inference and emit
#: exactly as many as will be executed. Both hold here: the reference is the VLA's own
#: chunk[:chunk_size] -- the plan the robot would otherwise have run -- and the actor
#: emits chunk_size * ACTION_DIM. Keeping one decision per executed chunk (stride ==
#: chunk_size, enforced in train.py) is what makes the first half true.
#:
#: "absolute" additionally requires conversion="plan_time", so it is now incompatible
#: with the measured default regime; it survives only to reproduce the discarded matrix.
DEFAULT_RL_ACTION_SPACE = "delta"


# ======================================================================================
# Run configurations
# ======================================================================================


def _check_scene(name: str) -> Scene:
    if name not in SCENES:
        raise KeyError(f"unknown scene {name!r}; known scenes: {sorted(SCENES)}")
    return SCENES[name]


def _check_conversion(mode: str) -> str:
    if mode not in ("plan_time", "step_time"):
        raise ValueError(f"conversion must be 'plan_time' or 'step_time', got {mode!r}")
    return mode


def _check_rl_spaces(action_space: str, conversion: str) -> str:
    """Empty string when the RL action space and the conversion regime agree."""
    if action_space not in ("absolute", "delta"):
        return f"rl_action_space must be 'absolute' or 'delta', got {action_space!r}"
    if action_space == "absolute" and conversion != "plan_time":
        return (
            "rl_action_space='absolute' requires conversion='plan_time'. The actor emits "
            "absolute joint targets, the corrector subtracts the arm state so the "
            "executor can add it back, and only plan_time adds back the same state the "
            "corrector subtracted. Under step_time the round trip is off by the distance "
            "travelled since the plan. Use rl_action_space='delta' with step_time."
        )
    return ""


@dataclass
class EvalConfig:
    """One evaluation run: the frozen VLA, or an RL actor on top of it."""

    # --- what to run on ---
    #: A scene name from SCENES, or "" to use the shipped val benchmark. Defaults to a
    #: selected scene; the inherited ones are superseded and have to be named explicitly.
    scene: str = "desk_mug"
    #: "eval" (the held-out half, and what every reported number must use) or "train"
    #: (the half RL learns on -- for token collection, never for a result).
    split: str = "eval"
    #: Run this benchmark directory instead of the one the scene names. For inspecting
    #: a candidate before it is promoted to a scene -- a candidate has no entry in SCENES
    #: yet, and adding one just to look at it would put an unmeasured scene in the config.
    benchmark_override: str = ""
    #: Write the frozen VLA's prefix tokens to this directory as phase-1 training data.
    #: Empty disables it. Turning it on forces the client to request tokens on every
    #: plan, which costs ~7.9 MB per call, so it is off unless a corpus is being built.
    record_tokens: str = ""
    #: Episode SPECS to run -- not rollouts. MolmoSpaces evaluates every spec several
    #: times and names the videos batch_i_of_n; measured, n = ceil(specs / 4) when all
    #: the specs share one house, which is exactly the case for a jittered scene. So
    #:
    #:     rollouts = specs * ceil(specs / 4)
    #:
    #: which is quadratic: 6 specs give 12 rollouts, 16 give 64, and 48 give 576. A
    #: 48-spec scene measurement was left running for three hours before that was
    #: noticed. Pick specs for the number of ROLLOUTS wanted: 16 specs is 64 rollouts,
    #: a +-12 point interval, and about 45 minutes on one GPU.
    #:
    #: The rate in result.json is over rollouts, so it is correct either way -- only the
    #: cost is surprising. `evaluate()` records batches_per_episode when it sees this.
    episodes: int = 16
    #: Simulator steps before an episode is cut off.
    horizon: int = 500

    # --- how the chunk is executed (see the block above) ---
    chunk_size: int = DEFAULT_CHUNK_SIZE
    conversion: str = DEFAULT_CONVERSION
    grasping: str = DEFAULT_GRASPING
    grasp_threshold: float = DEFAULT_GRASP_THRESHOLD

    # --- evaluating a trained actor; leave empty for the plain VLA ---
    actor: str = ""  #: checkpoint from run_train.py
    token_ae: str = ""  #: the phase-1 encoder that actor was trained with
    #: Env step at which RL takes over. 0 (the default) means the actor drives from the
    #: first step; set N to let the frozen VLA run the first N env steps. -1 uses the
    #: scene catalog gate (the old measured handover). Train and eval must match.
    #: Ignored when gate_frac > 0.
    gate_step: int = 0
    #: Fraction of `horizon` the frozen VLA runs before RL. 0.1 ≈ 10% of the
    #: episode (48 of 500, 40 of 400). 0 leaves `gate_step` in charge.
    gate_frac: float = GATE_FRAC
    gate_latch: bool = GATE_LATCH
    rl_action_space: str = DEFAULT_RL_ACTION_SPACE

    # --- machine ---
    checkpoint: Path = CHECKPOINT
    gpu: int = 0  #: CUDA device for the policy server
    sim_gpu: int = 0  #: CUDA device for the simulator
    egl_device: int = 0  #: MUJOCO_EGL_DEVICE_ID; EGL order is not CUDA order
    port: int = 8080
    workers: int = 1  #: one per server. Two clients on one server time out; measured.
    save_video: bool = True
    out_dir: Path = ROOT / "pi05_runs/eval"
    tag: str = ""  #: subdirectory name; defaults to a description of the run

    def scene_or_none(self) -> Scene | None:
        return _check_scene(self.scene) if self.scene else None

    def resolved_gate_step(self) -> int:
        """Env step at which RL takes over. 0 is step 0; -1 is the scene catalog."""
        return _resolve_gate_step(
            self.gate_step,
            self.scene_or_none(),
            gate_frac=self.gate_frac,
            horizon=self.horizon,
            chunk_size=self.chunk_size,
        )

    def benchmark_dir(self) -> Path:
        """Which repeats to run.

        Defaults to the held-out half. Reporting a number measured on `split="train"`
        would be reporting the episodes RL learned on, so the default is the safe one and
        the training half has to be asked for by name -- which token collection does.
        """
        if self.benchmark_override:
            return Path(self.benchmark_override)
        scene = self.scene_or_none()
        if scene is None:
            return VAL_BENCHMARK
        return scene.benchmark_train if self.split == "train" else scene.benchmark_eval

    def run_dir(self) -> Path:
        return Path(self.out_dir) / (self.tag or self.default_tag())

    def default_tag(self) -> str:
        who = "vla" if not self.actor else "actor"
        where = self.scene or "val"
        split = "" if not self.scene else f"_{self.split}"
        return f"{where}{split}_{who}_chunk{self.chunk_size}_{self.conversion}"

    def validate(self) -> str:
        """Empty string when the config is runnable, else the reason it is not."""
        if self.scene:
            _check_scene(self.scene)
        _check_conversion(self.conversion)
        if not 1 <= self.chunk_size <= VLA_CHUNK:
            return f"chunk_size must be in 1..{VLA_CHUNK}, got {self.chunk_size}"
        if self.split not in ("eval", "train"):
            return f"split must be 'eval' or 'train', got {self.split!r}"
        if self.split == "train" and not self.record_tokens:
            return (
                "split='train' runs the episodes RL learns on, so its success rate is "
                "not a result. Set record_tokens to build a corpus, or use split='eval'."
            )
        if self.grasping not in ("binary", "continuous"):
            return f"grasping must be 'binary' or 'continuous', got {self.grasping!r}"
        if self.workers != 1:
            return (
                "workers must be 1: pi0.5 inference blocks the server event loop, so a "
                "second client times out during the handshake and its episodes are "
                "silently skipped. Run one worker per server, one server per GPU."
            )
        if bool(self.actor) != bool(self.token_ae):
            return "actor and token_ae go together: an actor cannot run without its encoder"
        problem = _check_gate_frac(self.gate_frac)
        if problem:
            return problem
        if self.actor:
            problem = _check_rl_spaces(self.rl_action_space, self.conversion)
            if problem:
                return problem
            try:
                gate = self.resolved_gate_step()
            except ValueError as error:
                return str(error)
            if gate % self.chunk_size:
                return (
                    f"gate_step {gate} is not a multiple of chunk_size {self.chunk_size}; "
                    "the handover would slip to the next chunk boundary"
                )
        return ""


@dataclass
class RLConfig:
    """One RL-Token training run (phases 2-3, the online actor-critic).

    The hyperparameters the paper fixes live in rlt/config.py:OnlineConfig and are not
    duplicated here. This holds the ones that decide what the experiment means.
    """

    # --- what to run on ---
    scene: str = "desk_mug"
    encoder: str = "desk_mug"  #: which phase-1 autoencoder; see TOKEN_AE_DIR
    #: Benchmark episodes the collectors cycle through. The jittered scene benchmarks
    #: hold repeats numbered from 0; keep the tail unused for evaluation.
    episode_pool: str = "0-11"
    episodes: int = 300  #: total rollouts, warmup included
    warmup_episodes: int = 40  #: pure VLA; also measures the baseline in place
    horizon: int = 500

    # --- the execution regime. Must match the evaluation regime, or the result cannot
    # --- be read: the first matrix trained at 8 and evaluated at 1, so "RL helped" could
    # --- not be separated from "the chunk changed".
    chunk_size: int = DEFAULT_CHUNK_SIZE
    conversion: str = DEFAULT_CONVERSION
    grasping: str = DEFAULT_GRASPING
    grasp_threshold: float = DEFAULT_GRASP_THRESHOLD
    #: How often the VLA is queried and a decision recorded, in env steps. Must divide
    #: chunk_size so every chunk boundary is also a decision point. 0 means "same as
    #: chunk_size", which stores exactly one row per executed chunk and splices nothing.
    stride: int = 0

    # --- where RL takes over ---
    #: Env step at which RL takes over. 0 (the default) means the actor drives from the
    #: first env step — no frozen-VLA prefix. Set N to let the default policy run the
    #: first N steps, or -1 to use the scene catalog gate (mug 56, kettle 40, …).
    #: Ignored when gate_frac > 0.
    gate_step: int = 0
    #: Fraction of `horizon` the frozen VLA runs before RL. 0.1 ≈ 10% of the
    #: episode (PI05 default: 48 of 500, 40 of 400). 0 leaves `gate_step` in charge.
    gate_frac: float = GATE_FRAC
    gate_latch: bool = GATE_LATCH
    rl_action_space: str = DEFAULT_RL_ACTION_SPACE
    #: Whether transitions from before the gate opens enter the actor's replay buffer.
    #:
    #: DECIDED 2026-08-20: False. The split the project owner asked for is
    #:
    #:     phase-1 encoder   trained on the WHOLE episode, gate or no gate
    #:     actor / critic    only the steps the actor could have influenced
    #:
    #: and those are two different passes, so one flag cannot serve both. The encoder's
    #: corpus comes from a collection run (`EvalConfig.record_tokens`), which requests
    #: tokens on every plan from step 0 and is unaffected by this field. This field only
    #: governs the RL buffer, where pre-gate rows would spend update-to-data budget on
    #: states the actor never acts in.
    #:
    #: Leaving it False also lets a training rollout skip the 7.9 MB token transfer and
    #: the encoder pass entirely before the gate.
    store_pre_gate: bool = False

    # --- learning knobs worth seeing at this level; the rest are in rlt/config.py ---
    sigma: float = 0.02  #: exploration std of the Gaussian actor
    #: Pull toward the VLA reference chunk (paper Eq. 5). Default is the paper
    #: value 1.0. A 2026-08-21 desk_mug sweep found 100 tighter in delta space
    #: (deviation 67% vs 197% at 1.0); those runs are labeled beta=100. 1000
    #: collapses back to the frozen VLA.
    beta: float = 1.0
    gamma: float = 0.99  #: per env step; the bootstrap carries gamma^chunk_size
    utd: int = 5  #: update-to-data ratio, per stored transition
    seed: int = 0
    #: Empty means TOKEN_AE_DIR / ae_{encoder}.pt. Set this to train from a freshly
    #: collected corpus without replacing the original PI05 encoders.
    token_ae: str = ""
    #: Load a pretrained actor-critic (from run_pretrain_ac.py) before online starts.
    init_actor: str = ""
    #: Optional replay filled during that pretrain. Empty starts the online buffer empty.
    init_buffer: str = ""
    #: Continue a killed parallel run from actor_live.pt + buffer.npz + metrics.jsonl.
    #: Without this, train_parallel always rewrites episode.counter to 0 and reloads
    #: init_actor, which would throw away the online episodes already stored.
    resume: bool = False
    #: Actor-only episodes at the start of online (not stored). Seeds sr_last10.
    probe_episodes: int = 10
    #: Rollout processes on this GPU. 4 with egl_slots=3: three MuJoCo EGL
    #: contexts run and the fourth waits. They share one VLA server; /act is flocked.
    collectors: int = 4
    egl_slots: int = 3
    #: Env steps written to the online buffer (summed across workers) between
    #: actor-critic bursts. 0 = update on every absorbed row (sequential trainer).
    update_every_steps: int = 100
    algorithm: str = "rl_token"  # "rl_token" | "consensusflow" | "flow_rlt"
    #: flow_rlt only: RLTokenAgent checkpoint whose frozen critic scores the flow actor.
    rlt_critic: str = ""
    flow_actor_coef: float = 1.0  #: flow_rlt: weight on -Q(s, a), bounded in [0, 1]
    flow_bc_coef: float = 1.0  #: flow_rlt: weight on the flow-matching BC term
    #: flow_rlt with a learned critic: freeze it after the offline pretrain (arm 2)
    #: instead of continuing TD online (arm 3).
    flow_freeze_critic_online: bool = False
    #: flow_rlt CF composition: actor is a guidance G added to the analytic base
    #: velocity toward the reference (V = v_pi05_base + G), not a full corrector.
    flow_compose: bool = False
    train_token_offline: bool = False
    train_token_online: bool = False
    ae_finetune: bool = False
    init_random_ae: bool = False
    store_decision_tokens: bool = False
    record_tokens: str = ""
    token_replay: str = ""
    vla_traj: str = ""
    dump_vla_traj: str = ""

    # --- machine ---
    checkpoint: Path = CHECKPOINT
    gpu: int = 0
    egl_device: int = 0
    port: int = 8600
    out_dir: Path = ROOT / "pi05_runs/rl"
    tag: str = ""

    def resolved_stride(self) -> int:
        return self.stride or self.chunk_size

    def resolved_gate_step(self) -> int:
        """Env step at which RL takes over. 0 is step 0; -1 is the scene catalog."""
        return _resolve_gate_step(
            self.gate_step,
            self.scene_def(),
            gate_frac=self.gate_frac,
            horizon=self.horizon,
            chunk_size=self.chunk_size,
        )

    def scene_def(self) -> Scene:
        return _check_scene(self.scene)

    def benchmark_dir(self) -> Path:
        """Always the training half. The held-out half is what judges the actor, so the
        trainer is not given the option of learning on it."""
        return self.scene_def().benchmark_train

    def token_ae_path(self) -> Path:
        path = Path(self.token_ae) if self.token_ae else TOKEN_AE_DIR / f"ae_{self.encoder}.pt"
        if not path.exists():
            available = sorted(p.stem[3:] for p in TOKEN_AE_DIR.glob("ae_*.pt"))
            raise FileNotFoundError(f"no encoder at {path}; available: {available}")
        return path

    def run_dir(self) -> Path:
        return Path(self.out_dir) / (self.tag or self.default_tag())

    def default_tag(self) -> str:
        return f"{self.scene}_ae-{self.encoder}_chunk{self.chunk_size}_gate{self.resolved_gate_step()}"

    def validate(self) -> str:
        """Empty string when the config is runnable, else the reason it is not."""
        _check_scene(self.scene)
        _check_conversion(self.conversion)
        if not 1 <= self.chunk_size <= VLA_CHUNK:
            return f"chunk_size must be in 1..{VLA_CHUNK}, got {self.chunk_size}"
        stride = self.resolved_stride()
        if self.chunk_size % stride != 0:
            return (
                f"stride {stride} must divide chunk_size {self.chunk_size} so every chunk "
                "boundary stays a decision point"
            )
        if self.warmup_episodes < 0:
            return f"warmup_episodes must be >= 0, got {self.warmup_episodes}"
        if self.warmup_episodes >= self.episodes:
            return "warmup_episodes must be smaller than episodes"
        if self.gate_step < -1:
            return (
                f"gate_step must be >= -1 (0 = RL from step 0, -1 = scene catalog), "
                f"got {self.gate_step}"
            )
        problem = _check_gate_frac(self.gate_frac)
        if problem:
            return problem
        try:
            gate = self.resolved_gate_step()
        except ValueError as error:
            return str(error)
        if gate % self.chunk_size:
            return (
                f"gate_step {gate} is not a multiple of chunk_size {self.chunk_size}, so "
                "it falls inside a chunk. Decisions happen only at chunk boundaries, so "
                "the handover would silently slip to the next one."
            )
        return _check_rl_spaces(self.rl_action_space, self.conversion)


# ======================================================================================
# Passing a config to the simulator worker process
#
# MolmoSpaces builds the policy inside eval_main, which we do not control and which takes
# a "module:Class" string, not an object. The resolved config is therefore written to
# JSON and its path handed over in one environment variable, rather than a dozen.
# ======================================================================================

RUN_CONFIG_ENV = "PI05_RUN_CONFIG"


#: The torch device every run uses. It is always cuda:0 and never cuda:<gpu>, because
#: `prepare_environment` sets CUDA_VISIBLE_DEVICES to the single card the run owns -- so
#: inside the process that card IS index 0 and no other index exists. Building the string
#: from cfg.gpu instead killed all seven cells of a matrix wave on startup with
#: "Attempting to deserialize object on CUDA device 1 but device_count() is 1".
TORCH_DEVICE = "cuda:0"


def needs_encoder(cfg) -> bool:
    """Whether a run has to load the phase-1 encoder.

    Tokens get requested for two unrelated reasons: an RL run READS them through the
    encoder, a collection run only WRITES them down. Only the first needs an encoder.
    Treating "wants tokens" as "needs an encoder" made every collection episode die in
    torch.load on an empty path -- and because the episode runner swallows per-episode
    exceptions, the corpus stayed empty without a single error reaching the summary.
    """
    return bool(getattr(cfg, "actor", ""))


def apply_overrides(cfg, pairs):
    """Apply ``name=value`` strings to a config, coercing to the field's type.

    This exists for one-off sweeps from the shell. The intended way to change a setting
    is to edit its default above, so that the value is recorded in the repository rather
    than in somebody's shell history.
    """
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected name=value, got {pair!r}")
        name, _, raw = pair.partition("=")
        name = name.strip()
        field = cfg.__dataclass_fields__.get(name)
        if field is None:
            known = sorted(cfg.__dataclass_fields__)
            raise KeyError(f"{type(cfg).__name__} has no field {name!r}; known: {known}")
        setattr(cfg, name, _coerce(raw.strip(), field.type, name))
    return cfg


def _coerce(raw: str, annotation, name: str):
    text = str(annotation)
    if "bool" in text:
        lowered = raw.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name}: expected a boolean, got {raw!r}")
    # float before int: "int" is not a substring of "float", but keep the order so
    # gate_frac=0.1 cannot be parsed as an integer.
    if "float" in text:
        return float(raw)
    if "int" in text:
        return int(raw)
    if "Path" in text:
        return Path(raw)
    return raw


def _jsonable(value):
    return str(value) if isinstance(value, Path) else value


def save_run_config(cfg, path: Path) -> Path:
    """Write a resolved config next to the run it produced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: _jsonable(v) for k, v in asdict(cfg).items()}
    payload["__class__"] = type(cfg).__name__
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_run_config(path: str | Path | None = None):
    """Read back a config written by `save_run_config`.

    With no path, reads the one named by $PI05_RUN_CONFIG -- which is how the simulator
    worker process gets it.
    """
    if path is None:
        path = os.environ.get(RUN_CONFIG_ENV, "")
        if not path:
            raise RuntimeError(
                f"${RUN_CONFIG_ENV} is not set, so the worker process cannot tell which "
                "chunk size or execution regime it is meant to run. It is set by "
                "pi05/eval.py and pi05/train.py; a hand-rolled command has to set it too."
            )
    payload = json.loads(Path(path).read_text())
    kind = payload.pop("__class__", "EvalConfig")
    cls = {"EvalConfig": EvalConfig, "RLConfig": RLConfig}[kind]
    fields = set(cls.__dataclass_fields__)
    known = {k: v for k, v in payload.items() if k in fields}
    for name in ("checkpoint", "out_dir"):
        if name in known:
            known[name] = Path(known[name])
    return cls(**known)


# ======================================================================================


def describe() -> str:
    """Human-readable summary of the current settings. Printed by every entry point."""
    lines = ["scenes:"]
    for scene in SCENES.values():
        lines.append(f"  {scene.name:9s} val episode {scene.val_episode:>3}  {scene.task}")
        for piece in _wrap(scene.note, 74):
            lines.append(f"            {piece}")
    lines += [
        "",
        f"checkpoint      {CHECKPOINT}",
        f"encoders        {TOKEN_AE_DIR}",
        f"val benchmark   {VAL_BENCHMARK}",
        "",
        f"VLA predicts    {VLA_CHUNK} actions per call, {ACTION_DIM} dims each",
        f"executed        {DEFAULT_CHUNK_SIZE} per plan by default ({DEFAULT_CONVERSION})",
        f"token width     {VLA_TOKEN_DIM}  (export RLT_VLA_TOKEN_DIM=2048 for RL runs)",
        f"RL gate         step 0 (set gate_step=N or gate_frac=0.1 for a ~10% VLA prefix)",
        f"RL refines      {DEFAULT_RL_ACTION_SPACE} actions",
    ]
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    print(describe())
