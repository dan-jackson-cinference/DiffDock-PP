import glob
import os
from typing import Optional

from torch import nn
from torch_geometric.nn.data_parallel import DataParallel

from config import DiffusionCfg, E3NNCfg
from geom_utils.transform import NoiseSchedule
from model.diffusion import TensorProductScoreModel
from model.losses import DiffusionLoss
from utils import get_model_path

from .model import ConfidenceModel, ScoreModel


def load_score_model(
    diffusion_cfg: DiffusionCfg,
    e3nn_cfg: E3NNCfg,
    noise_schedule: NoiseSchedule,
    score_model_path: Optional[str],
    dropout: float,
    num_atoms: int,
    num_gpu: int,
    fold: int,
    load_best: bool = True,
    checkpoint_path: Optional[str] = None,
) -> ScoreModel:
    """
    Model factory
    :load_best: if True, load best model in terms of performance on val set, else load last model
    """

    encoder = TensorProductScoreModel.from_config(
        e3nn_cfg, noise_schedule, dropout, num_atoms, confidence_mode=False
    )
    loss = DiffusionLoss.from_config(diffusion_cfg, noise_schedule, num_gpu)
    model = ScoreModel(loss, encoder)
    # (optional) load checkpoint if provided
    checkpoint = get_checkpoint(score_model_path, checkpoint_path, fold, load_best)
    model.load_checkpoint(checkpoint)
    return model


def load_confidence_model(
    e3nn_cfg: E3NNCfg,
    noise_schedule: NoiseSchedule,
    filtering_model_path: Optional[str],
    dropout: float,
    num_atoms: int,
    fold: int,
    load_best: bool = True,
    checkpoint_path: Optional[str] = None,
) -> ConfidenceModel:
    # load model with specified arguments
    encoder = TensorProductScoreModel.from_config(
        e3nn_cfg, noise_schedule, dropout, num_atoms, confidence_mode=True
    )
    model = ConfidenceModel(encoder)
    checkpoint = get_checkpoint(filtering_model_path, checkpoint_path, fold, load_best)
    model.load_checkpoint(checkpoint)

    return model


def get_checkpoint(
    model_path: Optional[str],
    checkpoint_path: Optional[str],
    fold: int,
    load_best: bool,
) -> Optional[str]:
    checkpoint = None
    # (optional) load checkpoint if provided
    if model_path is not None:
        load_best = True  # TODO: Remove
        if load_best:
            checkpoint = select_model(model_path, True)
        else:
            checkpoint = os.path.join(model_path, "model_last.pth")
    elif checkpoint_path is not None:
        fold_dir = os.path.join(checkpoint_path, f"fold_{fold}")
        if load_best:
            checkpoint = get_model_path(fold_dir)
        else:
            checkpoint = os.path.join(fold_dir, "model_last.pth")

    return checkpoint


def select_model(fold_dir, confidence_mode) -> str:
    paths = []
    for path in glob.glob(f"{fold_dir}/*.pth"):
        if "last" not in path:
            paths.append(path)

    if confidence_mode:
        models = sorted(
            paths, key=lambda s: -float(s.split("/")[-1].split("_")[-1][:-4])
        )
    else:
        models = sorted(paths, key=lambda s: float(s.split("/")[-1].split("_")[4]))

    if len(models) == 0:
        print(f"no models found at {fold_dir}")
        return
    checkpoint = models[0]
    return checkpoint


def to_cuda(model: ScoreModel, gpu: int, num_gpu: int) -> ScoreModel:
    """
    move model to cuda
    """
    # specify number in case test =/= train GPU
    if gpu >= 0:
        model = model.cuda(gpu)
        if num_gpu > 1:
            device_ids = [gpu + i for i in range(num_gpu)]
            model = DataParallel(model, device_ids=device_ids)
    return model
