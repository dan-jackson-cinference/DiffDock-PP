"""
This file has been heavily modified, mostly in terms
of syntax and nicer flow.

2022.11.08
"""
from __future__ import annotations
from typing import Optional

import copy
from dataclasses import dataclass


import numpy as np
import torch
from torch import Tensor, device
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from config import DiffusionCfg

from .geometry import axis_angle_to_matrix, kabsch_torch
from .so3 import sample_vec, score_vec
from .torus import score as score_torus


# ------ PyG UTILS -------
@dataclass
class NoiseSchedule:
    """
    Transforms t into scaled sigmas
    """

    tr_s_min: float
    tr_s_max: float
    rot_s_min: float
    rot_s_max: float
    tor_s_min: float
    tor_s_max: float

    @property
    def tr_scale(self) -> torch.Tensor:
        return torch.sqrt(2 * torch.log(torch.tensor(self.tr_s_max / self.tr_s_min)))

    @property
    def rot_scale(self) -> torch.Tensor:
        return torch.sqrt(torch.log(torch.tensor(self.rot_s_max / self.rot_s_min)))

    def __call__(
        self, t_tr: float, t_rot: float, t_tor: float
    ) -> tuple[float, float, float]:
        """
        Convert from time to (scaled) sigma space
        """
        tr_s = self.tr_s_min ** (1 - t_tr) * self.tr_s_max**t_tr
        rot_s = self.rot_s_min ** (1 - t_rot) * self.rot_s_max**t_rot
        tor_s = self.tor_s_min ** (1 - t_tor) * self.tor_s_max**t_tor
        return tr_s, rot_s, tor_s

    @classmethod
    def from_config(cls, cfg: DiffusionCfg) -> NoiseSchedule:
        return cls(
            cfg.tr_s_min,
            cfg.tr_s_max,
            cfg.rot_s_min,
            cfg.rot_s_max,
            cfg.tor_s_min,
            cfg.tor_s_max,
        )


class NoiseTransform(BaseTransform):
    """
    Apply translation, rotation, torsional noise
    """

    def __init__(self, noise_schedule: NoiseSchedule):
        # save min/max sigma scales
        self.noise_schedule = noise_schedule

        self.no_torsion = True  # >>> TODO

    def __call__(self, data):
        """
        Modifies data in place
        @param (torch_geometric.data.HeteroData) data
        """
        t = np.random.uniform()  # sample time
        t_tr, t_rot, t_tor = t, t, t  # same time scale for each
        data = self.apply_noise(data, t_tr, t_rot, t_tor)
        return data

    def apply_noise(
        self,
        data,
        t_tr: float,
        t_rot: float,
        t_tor: float,
        tr_update=None,
        rot_update=None,
        tor_updates=None,
    ):
        """
        Apply noise to existing HeteroData object
        @param (torch_geometric.data.HeteroData) data
        """
        tr_s, rot_s, tor_s = self.noise_schedule(t_tr, t_rot, t_tor)
        set_time(data, t_tr, t_rot, t_tor, 1, device=None)
        # sample updates if not provided
        if tr_update is None:
            tr_update = torch.normal(mean=0, std=tr_s, size=(1, 3))
        if rot_update is None:
            rot_update = sample_vec(eps=rot_s)
            rot_update = torch.from_numpy(rot_update).float()
        if tor_updates is None and (not self.no_torsion):
            tor_updates = np.random.normal(
                loc=0.0, scale=tor_s, size=data["ligand"].edge_mask.sum()
            )

        # apply updates
        self.apply_updates(data, tr_update, rot_update, tor_updates)

        # compute ground truth score given updates, noise level
        self.get_score(data, tr_update, tr_s, rot_update, rot_s, tor_updates, tor_s)

        return data

    def apply_updates(
        self,
        data: HeteroData,
        tr_update: Tensor,
        rot_update: Tensor,
        tor_updates: Optional[Tensor] = None,
    ) -> HeteroData:
        """
        Apply translation / rotation / torsion updates to
        data["ligand"].pos
        @param (torch_geometric.data.HeteroData) data
        """
        com = torch.mean(data["ligand"].pos, dim=0, keepdim=True)
        rot_mat = axis_angle_to_matrix(rot_update.squeeze())
        rigid_new_pos = (data["ligand"].pos - com) @ rot_mat.T + tr_update + com

        if tor_updates is not None:
            # select edges to modify (torsion angles only)
            edge_index = data["ligand", "ligand"].edge_index
            edge_index = edge_index.T[data["ligand"].edge_mask]
            # mask which to update
            mask_rotate = data["ligand"].mask_rotate
            # data type
            as_numpy = isinstance(mask_rotate, np.ndarray)
            if not as_numpy:
                as_numpy = mask_rotate[0]
            flex_new_pos = self.apply_torsion_updates(
                rigid_new_pos, edge_index, mask_rotate, tor_updates, as_numpy
            )
            flex_new_pos = flex_new_pos.to(rigid_new_pos.device)
            # fix orientation to disentangle torsion update
            R, t = kabsch_torch(flex_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flex_new_pos @ R.T + t.T
            data["ligand"].pos = aligned_flexible_pos
        else:
            data["ligand"].pos = rigid_new_pos
        return data

    def apply_torsion_updates(
        pos, edge_index, mask_rotate, tor_updates, as_numpy=False
    ):
        """
        UNUSED as of now
        """
        pos = copy.deepcopy(pos)
        if type(pos) != np.ndarray:
            pos = pos.cpu().numpy()

        for idx_edge, e in enumerate(edge_index.cpu().numpy()):
            if tor_updates[idx_edge] == 0:
                continue
            u, v = e[0], e[1]

            # check if need to reverse the edge, v should be connected to the part that gets rotated
            assert not mask_rotate[idx_edge, u]
            assert mask_rotate[idx_edge, v]

            rot_vec = (
                pos[u] - pos[v]
            )  # convention: positive rotation if pointing inwards
            rot_vec = (
                rot_vec * tor_updates[idx_edge] / np.linalg.norm(rot_vec)
            )  # idx_edge!
            rot_mat = R.from_rotvec(rot_vec).as_matrix()

            pos[mask_rotate[idx_edge]] = (
                pos[mask_rotate[idx_edge]] - pos[v]
            ) @ rot_mat.T + pos[v]

        if not as_numpy:
            pos = torch.from_numpy(pos.astype(np.float32))
        return pos

    def get_score(self, data, tr_update, tr_s, rot_update, rot_s, tor_updates, tor_s):
        """
        Compute ground truth score, given updates and noise.
        Modifies data in place.
        """
        # translation score
        data.tr_score = -tr_update / tr_s**2
        # rotation score
        rot_score = score_vec(vec=rot_update, eps=rot_s)
        rot_score = rot_score.unsqueeze(0)
        data.rot_score = rot_score
        # torsion score
        if self.no_torsion:
            data.tor_score = None
            data.tor_s_edge = None
        else:
            tor_score = score_torus(tor_updates, tor_s)
            data.tor_score = torch.from_numpy(tor_score).float()
            tor_s_edge = np.ones(data["ligand"].edge_mask.sum())
            data.tor_s_edge = tor_s_edge * tor_s
        return data


def set_time(
    graph: HeteroData,
    t_tr: float,
    t_rot: float,
    t_tor: float,
    batch_size: int,
    device: Optional[device] = None,
) -> None:
    """
    Save sampled time to current batch
    """
    lig_size = graph["ligand"].num_nodes
    graph["ligand"].node_t = {
        "tr": t_tr * torch.ones(lig_size).to(device),
        "rot": t_rot * torch.ones(lig_size).to(device),
        "tor": t_tor * torch.ones(lig_size).to(device),
    }
    rec_size = graph["receptor"].num_nodes
    graph["receptor"].node_t = {
        "tr": t_tr * torch.ones(rec_size).to(device),
        "rot": t_rot * torch.ones(rec_size).to(device),
        "tor": t_tor * torch.ones(rec_size).to(device),
    }
    graph.complex_t = {
        "tr": t_tr * torch.ones(batch_size).to(device),
        "rot": t_rot * torch.ones(batch_size).to(device),
        "tor": t_tor * torch.ones(batch_size).to(device),
    }
