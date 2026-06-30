import os
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
from PIL import Image, ImageEnhance
from tensorboardX import SummaryWriter


def _is_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if _is_scalar(v)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if _is_scalar(v)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if _is_scalar(v)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


class TensorboardLogger:
    """Local TensorBoard logger."""

    def __init__(self, log_dir, hparams=None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        if hparams is not None:
            scalar_hparams = {
                k: v for k, v in hparams.items() if _is_scalar(v) and v is not None
            }
            if scalar_hparams:
                self.writer.add_hparams(scalar_hparams, {k: 0.0 for k in scalar_hparams})

    def log(self, metrics, step):
        for key, value in metrics.items():
            if _is_scalar(value):
                self.writer.add_scalar(key, float(value), step)
            elif isinstance(value, np.ndarray) and value.ndim == 4:
                self.log_video(key, value, step)

    def log_video(self, tag, video, step, fps=15):
        """Log a video array with shape (T, C, H, W)."""
        if video.ndim != 4:
            raise ValueError(f'Expected video shape (T, C, H, W), got {video.shape}')
        self.writer.add_video(tag, video[np.newaxis, ...], step, fps=fps)

    def close(self):
        self.writer.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_experiment_logging(save_dir, project, run_group, exp_name, hparams=None):
    """Create experiment directory and local TensorBoard logger."""
    exp_dir = os.path.join(save_dir, project, run_group, exp_name)
    tb_dir = os.path.join(exp_dir, 'tensorboard')
    os.makedirs(exp_dir, exist_ok=True)
    logger = TensorboardLogger(tb_dir, hparams=hparams)
    return logger, exp_dir


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def prepare_eval_video(renders=None, n_cols=None):
    """Prepare evaluation renders as a TensorBoard video array (T, C, H, W)."""
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)

    return reshape_video(renders, n_cols)
