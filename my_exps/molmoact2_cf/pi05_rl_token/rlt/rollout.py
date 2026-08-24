"""One benchmark episode at a time, with the policy kept between episodes.

`run_evaluation` is the only way to drive a MolmoSpaces episode, and with
`num_workers=1` it runs in-process, so the preloaded policy survives the call
and can be drained afterwards. Everything it writes to disk (HDF5 trajectories)
is scratch for us and is deleted per episode.
"""

from __future__ import annotations

import itertools
import logging
import os
import shutil
from pathlib import Path

from molmo_spaces.evaluation.eval_main import run_evaluation

from .eval_config import RLTokenEvalConfig, default_benchmark_dir
from .replay import Rollout

log = logging.getLogger("rlt.rollout")

_RUNNERS = itertools.count()  # so two runners never share a scratch directory


class EpisodeRunner:
    def __init__(
        self,
        policy,
        benchmark_dir: str,
        horizon: int,
        tmp_dir: str,
        video_dir: str = "",
        camera: str = "exo_camera_1",
        eval_config_cls=None,
    ) -> None:
        # Which MolmoSpaces experiment config the episode runs under. Defaults to the
        # MolmoAct2 one this package was built for; the pi0.5 path passes its own, which
        # is what selects the policy class and therefore the whole action path.
        self.eval_config_cls = eval_config_cls or RLTokenEvalConfig
        self.policy = policy
        self.benchmark_dir = Path(benchmark_dir) if benchmark_dir else default_benchmark_dir()
        self.horizon = horizon
        # Episode directories are numbered from 1 per runner and deleted after
        # use, so two runners pointed at the same scratch root would delete each
        # other's output -- which is how parallel eval shards lost their videos.
        self.tmp_dir = Path(tmp_dir) / f"pid{os.getpid()}_{next(_RUNNERS)}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        # MolmoSpaces writes one mp4 per episode per camera into the output dir,
        # which we otherwise delete. With video_dir set they are kept, named for
        # their outcome so a grid can be bordered without a results file.
        self.video_dir = Path(video_dir) if video_dir else None
        self.camera = camera
        if self.video_dir:
            self.video_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def _keep_video(self, episode_dir: Path, success: bool) -> None:
        source = next(episode_dir.glob(f"**/episode_*_{self.camera}*.mp4"), None)
        if source is None:
            log.warning("no %s video under %s", self.camera, episode_dir)
            return
        name = f"rollout_{self.count:03d}_{'ok' if success else 'fail'}_{self.camera}.mp4"
        shutil.copy2(source, self.video_dir / name)

    def run(self, episode_idx: int) -> Rollout | None:
        """One episode. Returns None if the environment failed to produce one."""
        self.count += 1
        episode_dir = self.tmp_dir / f"ep_{self.count:06d}"
        shutil.rmtree(episode_dir, ignore_errors=True)
        valid = success = False  # the finally block reads both

        try:
            results = run_evaluation(
                eval_config_cls=self.eval_config_cls,
                benchmark_dir=self.benchmark_dir,
                task_horizon_steps=self.horizon,
                num_workers=1,
                use_wandb=False,
                preloaded_policy=self.policy,
                episode_idx=episode_idx,
                output_dir=episode_dir,
            )
            valid = results.total_count > 0
            success = results.success_count > 0
        except Exception as error:  # noqa: BLE001
            if self.policy.fatal_error is not None:
                raise self.policy.fatal_error
            log.warning("episode %d failed inside the environment: %s", episode_idx, error)
            valid, success = False, False
        finally:
            if self.video_dir and valid:
                self._keep_video(episode_dir, success)
            shutil.rmtree(episode_dir, ignore_errors=True)

        if self.policy.fatal_error is not None:
            raise self.policy.fatal_error
        if not valid or self.policy.step == 0:
            self.policy.reset()
            return None
        return self.policy.pop_rollout(success)
