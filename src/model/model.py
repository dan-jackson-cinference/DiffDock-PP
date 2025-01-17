from typing import Optional

import torch
from torch import Tensor, nn

from utils import printt

from .diffusion import TensorProductScoreModel
from .losses import DiffusionLoss


class BaseModel(nn.Module):
    """
    enc(receptor) -> R^(dxL)
    enc(ligand)  -> R^(dxL)
    """

    def __init__(self, encoder: TensorProductScoreModel):
        super(BaseModel, self).__init__()

        ######## initialize (shared) modules
        # raw encoders
        self.encoder = encoder

        self._init()

    def _init(self):
        for name, param in self.named_parameters():
            # NOTE must name parameter "bert"
            if "bert" in name:
                continue
            # bias terms
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            # weight terms
            else:
                nn.init.xavier_normal_(param)

    def dist(self, x, y):
        if len(x.size()) > 1:
            return ((x - y) ** 2).sum(-1)
        return (x - y) ** 2

    def load_checkpoint(self, checkpoint: Optional[str]) -> None:
        if checkpoint is not None:
            # extract current model
            state_dict = self.state_dict()
            # load onto CPU, transfer to proper GPU
            pretrain_dict = torch.load(checkpoint, map_location="cpu")["model"]
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in state_dict}
            # update current model
            state_dict.update(pretrain_dict)
            # >>>
            for k, v in state_dict.items():
                if k not in pretrain_dict:
                    print(k, "not saved")
            self.load_state_dict(state_dict)
            printt("loaded checkpoint from", checkpoint)
        else:
            printt("no checkpoint found")


class ScoreModel(BaseModel):
    def __init__(self, loss: DiffusionLoss, encoder: TensorProductScoreModel):
        super().__init__(encoder)
        # loss function
        self.loss = loss

        self._init()

    def forward(self, batch) -> dict[str, Tensor]:
        # move graphs to cuda
        tr_pred, rot_pred, tor_pred = self.encoder(batch)

        outputs: dict[str, Tensor] = {}
        outputs["tr_pred"] = tr_pred
        outputs["rot_pred"] = rot_pred
        outputs["tor_pred"] = tor_pred

        return outputs

    def compute_loss(self, batch, outputs):
        losses = self.loss(batch, outputs)
        return losses


class ConfidenceModel(BaseModel):
    def __init__(
        self,
        encoder: TensorProductScoreModel,
    ):
        super().__init__(encoder)

        self._init()

    def forward(self, batch):
        # move graphs to cuda
        logits = self.encoder(batch)

        return logits
