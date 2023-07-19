"""
    Inference script
"""

import copy
import os
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch import Tensor, device
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataListLoader, DataLoader

from config import TempCfg
from data.protein import PPComplex
from data.dataset import BindingDataset, SamplingDataset
from geom_utils import set_time
from geom_utils.transform import NoiseTransform
from model.model import ScoreModel
from utils import printt


def sample(
    data: dict[str, PPComplex],
    score_model: ScoreModel,
    num_steps: int,
    no_torsion: bool,
    tr_s_max: float,
    noise_transform: NoiseTransform,
    epoch: int = 0,
    num_gpu: int = 1,
    gpu: int = 0,
    visualize_first_n_samples=0,
    visualization_dir="./visualization",
    in_batch_size=None,
):
    """
    Run reverse process
    """
    if in_batch_size is None:
        in_batch_size = args.batch_size
    # switch to eval mode
    score_model.eval()

    # diffusion timesteps
    timesteps = get_timesteps(num_steps)

    # Prepare for visualizations
    visualize_first_n_samples = min(visualize_first_n_samples, len(data_list))
    graph_gts = [data_list[i] for i in range(visualize_first_n_samples)]
    visualization_values = [
        data_list.get_visualization_values(index=i)
        for i in range(visualize_first_n_samples)
    ]
    four_letter_pdb_names = [
        get_four_letters_pdb_identifier(graph_gt.name) for graph_gt in graph_gts
    ]
    visualization_dirs = create_visualization_directories(
        visualization_dir, epoch, four_letter_pdb_names
    )

    # randomize original position and COPY data_list
    data_list = randomize_position(data, tr_s_max, no_torsion)

    # For visualization
    for i in range(visualize_first_n_samples):
        write_pdb(
            visualization_values[i],
            graph_gts[i],
            "receptor",
            f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-receptor.pdb",
        )
        write_pdb(
            visualization_values[i],
            graph_gts[i],
            "ligand",
            f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-gt.pdb",
        )
        write_pdb(
            visualization_values[i],
            data_list[i],
            "ligand",
            f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-0.pdb",
        )

    # sample
    for t_idx in range(num_steps):
        # create new loader with current step graphs
        if torch.cuda.is_available() and num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader
        test_loader = loader(data_list, batch_size=args.batch_size)
        new_data_list = []  # updated every step
        # DiffDock uses same schedule for all noise
        cur_t = timesteps[t_idx]
        if t_idx == num_steps - 1:
            dt = cur_t
        else:
            dt = cur_t - timesteps[t_idx + 1]

        for com_idx, complex_graphs in enumerate(test_loader):
            # move to CUDA
            # complex_graphs = complex_graphs.cuda()
            if torch.cuda.is_available() and num_gpu == 1:
                complex_graphs = complex_graphs.cuda(gpu)

            # this MAY differ from args.batch_size
            # based on # GPUs and last batch
            if type(complex_graphs) is list:
                batch_size = len(complex_graphs)
            else:
                batch_size = complex_graphs.num_graphs

            # convert to sigma space and save time
            tr_s, rot_s, tor_s = noise_transform.noise_schedule(cur_t, cur_t, cur_t)
            device_for_set_time = (
                complex_graphs["ligand"]["pos"].device
                if torch.cuda.is_available() and num_gpu == 1
                else None
            )
            if type(complex_graphs) is list:
                for g in complex_graphs:
                    set_time(g, cur_t, cur_t, cur_t, 1, device_for_set_time)
            else:
                set_time(
                    complex_graphs, cur_t, cur_t, cur_t, batch_size, device_for_set_time
                )

            with torch.no_grad():
                outputs = score_model(complex_graphs)
            tr_score = outputs["tr_pred"].cpu()
            rot_score = outputs["rot_pred"].cpu()
            tor_score = outputs["tor_pred"].cpu()

            # translation gradient (?)
            tr_g = tr_s * noise_transform.noise_schedule.tr_scale

            # rotation gradient (?)
            rot_g = 2 * rot_s * noise_transform.noise_schedule.rot_scale

            # actual update
            if args.ode:
                tr_update = 0.5 * tr_g**2 * dt * tr_score
                rot_update = 0.5 * rot_score * dt * rot_g**2
            else:
                if args.no_final_noise and t_idx == num_steps - 1:
                    tr_z = torch.zeros((batch_size, 3))
                    rot_z = torch.zeros((batch_size, 3))
                elif args.no_random:
                    tr_z = torch.zeros((batch_size, 3))
                    rot_z = torch.zeros((batch_size, 3))
                else:
                    tr_z = torch.normal(0, 1, size=(batch_size, 3))
                    rot_z = torch.normal(0, 1, size=(batch_size, 3))

                tr_update = tr_g**2 * dt * tr_score
                tr_update = tr_update + (tr_g * np.sqrt(dt) * tr_z)

                rot_update = rot_score * dt * rot_g**2
                rot_update = rot_update + (rot_g * np.sqrt(dt) * rot_z)

            if args.temp_sampling != 1.0:
                tr_sigma_data = np.exp(
                    args.temp_sigma_data_tr * np.log(args.tr_s_max)
                    + (1 - args.temp_sigma_data_tr) * np.log(args.tr_s_min)
                )
                lambda_tr = (tr_sigma_data + tr_s) / (
                    tr_sigma_data + tr_s / args.temp_sampling
                )
                tr_update = (
                    tr_g**2
                    * dt
                    * (lambda_tr + args.temp_sampling * args.temp_psi / 2)
                    * tr_score.cpu()
                    + tr_g * np.sqrt(dt * (1 + args.temp_psi)) * tr_z
                ).cpu()

                rot_sigma_data = np.exp(
                    args.temp_sigma_data_rot * np.log(args.rot_s_max)
                    + (1 - args.temp_sigma_data_rot) * np.log(args.rot_s_min)
                )
                lambda_rot = (rot_sigma_data + rot_s) / (
                    rot_sigma_data + rot_s / args.temp_sampling
                )
                rot_update = (
                    rot_g**2
                    * dt
                    * (lambda_rot + args.temp_sampling * args.temp_psi / 2)
                    * rot_score.cpu()
                    + rot_g * np.sqrt(dt * (1 + args.temp_psi)) * rot_z
                ).cpu()

            # apply transformations
            if not isinstance(complex_graphs, list):
                complex_graphs = complex_graphs.to("cpu").to_data_list()
            for i, data in enumerate(complex_graphs):
                new_graph = noise_transform.apply_updates(
                    data, tr_update[i : i + 1], rot_update[i : i + 1].squeeze(0), None
                )

                new_data_list.append(new_graph)
            # === end of batch ===
            # printt(f'finished batch {com_idx}')

        for i in range(visualize_first_n_samples):
            write_pdb(
                visualization_values[i],
                new_data_list[i],
                "ligand",
                f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-{t_idx + 1}.pdb",
            )

        # update starting point for next step
        assert len(new_data_list) == len(data_list)
        data_list = new_data_list
        printt(f"Completed {t_idx} out of {num_steps} steps")

        # Cut last diffusion steps short because they tend to overfit
        if t_idx >= args.actual_steps - 1:
            break
        # === end of timestep ===

    return data_list  # , batch_size_to_return


def create_visualization_directories(
    top_visualization_dir: str, epoch: int, pdb_names: list[str]
):
    visualization_dirs = [
        f"{top_visualization_dir}/epoch-{epoch}/{pdb_name}" for pdb_name in pdb_names
    ]
    for directory in visualization_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return visualization_dirs


def get_timesteps(inference_steps: int):
    return np.linspace(1, 0, inference_steps + 1)[:-1]


def initialize_random_positions(
    graph: HeteroData, num_trajectories: int, tr_s_max: float, no_torsion: bool = True
) -> list[HeteroData]:
    """
    Modify COPY of data_list objects
    """

    if not no_torsion:
        raise Exception("not yet implemented")
        # randomize torsion angles
        for i, complex_graph in enumerate(data_list):
            torsion_updates = np.random.uniform(
                low=-np.pi, high=np.pi, size=complex_graph["ligand"].edge_mask.sum()
            )
            complex_graph["ligand"].pos = modify_conformer_torsion_angles(
                complex_graph["ligand"].pos,
                complex_graph["ligand", "ligand"].edge_index.T[
                    complex_graph["ligand"].edge_mask
                ],
                complex_graph["ligand"].mask_rotate[0],
                torsion_updates,
            )
            data_list.set_graph(i, complex_graph)

    graph_init_positions: list[HeteroData] = []
    for i in range(num_trajectories):
        graph_copy = copy.deepcopy(graph)

        pos = graph_copy["ligand"].pos
        center = torch.mean(pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix())
        pos = (pos - center) @ random_rotation.T.float()

        # random translation
        tr_update = torch.normal(0, tr_s_max, size=(1, 3))
        pos = pos + tr_update
        graph_copy["ligand"].pos = pos
        graph_init_positions.append(graph_copy)

    return graph_init_positions


class Sampler:
    def __init__(
        self,
        score_model: ScoreModel,
        noise_transform: NoiseTransform,
        num_steps: int,
        batch_size: int,
        no_final_noise: bool,
        no_random: bool,
        device: device,
    ):
        self.score_model = score_model
        self.noise_transform = noise_transform
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.no_final_noise = no_final_noise
        self.no_random = no_random
        self.device = device

    def sample(
        self,
        initial_positions: SamplingDataset,
        ode: bool,
        temp_cfg: Optional[TempCfg] = None,
    ) -> list[SamplingDataset]:
        time_steps = self.get_time_steps()
        samples: list[SamplingDataset] = [initial_positions]

        for t_idx in tqdm(range(self.num_steps)):
            cur_t = time_steps[t_idx]
            if t_idx == self.num_steps - 1:
                dt = cur_t
            else:
                dt = cur_t - time_steps[t_idx + 1]

            new_samples = self.sample_step(cur_t, dt, t_idx, samples[-1], ode, temp_cfg)
            # print(new_samples)
            samples.append(SamplingDataset(new_samples))

        return samples

    def sample_step(
        self,
        t: float,
        dt: float,
        idx: int,
        graph_dataset: SamplingDataset,
        ode: bool,
        temp_cfg: Optional[TempCfg],
    ) -> list[HeteroData]:
        for graph in graph_dataset:
            set_time(graph, t, t, t, 1, self.device)

        dataloader = DataLoader(
            graph_dataset, batch_size=self.batch_size, shuffle=False
        )

        all_new_graphs: list[HeteroData] = []
        for batch_of_graphs in dataloader:
            batch_of_graphs = batch_of_graphs.cuda(0)
            with torch.no_grad():
                outputs = self.score_model(batch_of_graphs)
            tr_score = outputs["tr_pred"].cpu()
            rot_score = outputs["rot_pred"].cpu()
            tor_score = outputs["tor_pred"].cpu()

            # translation gradient (?)
            tr_s, rot_s, tor_s = self.noise_transform.noise_schedule(t, t, t)
            tr_g = tr_s * self.noise_transform.noise_schedule.tr_scale

            # rotation gradient (?)
            rot_g = 2 * rot_s * self.noise_transform.noise_schedule.rot_scale

            tr_z, rot_z = self.trz_rotz(idx)
            if ode:
                tr_update, rot_update = self.ode_update(
                    dt, tr_g, rot_g, tr_score, rot_score
                )
            elif temp_cfg is not None:
                tr_update, rot_update = self.sample_temp(
                    dt,
                    tr_s,
                    tr_g,
                    tr_z,
                    tr_score,
                    rot_s,
                    rot_g,
                    rot_z,
                    rot_score,
                    temp_cfg.temp_sampling,
                    temp_cfg.temp_psi,
                    temp_cfg.temp_sigma_data_tr,
                    temp_cfg.temp_sigma_data_rot,
                )
            else:
                tr_update, rot_update = self.update(
                    dt, tr_z, rot_z, tr_g, rot_g, tr_score, rot_score
                )

            # apply transformations
            graph_list = batch_of_graphs.to("cpu").to_data_list()
            new_graphs = [
                self.noise_transform.apply_updates(
                    data, tr_update[i : i + 1], rot_update[i : i + 1].squeeze(0), None
                )
                for i, data in enumerate(graph_list)
            ]
            all_new_graphs.extend(new_graphs)
        # new_graph = self.noise_transform.apply_updates(
        #     graph_dataset, tr_update, rot_update.squeeze(0)
        # )
        return all_new_graphs

    def get_time_steps(self):
        return np.linspace(1, 0, self.num_steps + 1)[:-1]

    def trz_rotz(self, idx: int) -> tuple[Tensor, Tensor]:
        if (self.no_final_noise and idx == self.num_steps - 1) or self.no_random:
            return torch.zeros((self.batch_size, 3)), torch.zeros((self.batch_size, 3))

        return torch.normal(0, 1, size=(self.batch_size, 3)), torch.normal(
            0, 1, size=(self.batch_size, 3)
        )

    def ode_update(
        self,
        dt: float,
        tr_g: Tensor,
        rot_g: Tensor,
        tr_score: Tensor,
        rot_score: Tensor,
    ):
        tr_update = 0.5 * tr_g**2 * dt * tr_score
        rot_update = 0.5 * rot_score * dt * rot_g**2
        return tr_update, rot_update

    def update(
        self,
        dt: float,
        tr_z: Tensor,
        rot_z: Tensor,
        tr_g: Tensor,
        rot_g: Tensor,
        tr_score: Tensor,
        rot_score: Tensor,
    ):
        tr_update = tr_g**2 * dt * tr_score + (tr_g * np.sqrt(dt) * tr_z)
        rot_update = rot_score * dt * rot_g**2 + (rot_g * np.sqrt(dt) * rot_z)

        return tr_update, rot_update

    def sample_temp(
        self,
        dt: float,
        tr_s: float,
        tr_g: Tensor,
        tr_z: Tensor,
        tr_score: Tensor,
        rot_s: float,
        rot_g: Tensor,
        rot_z: Tensor,
        rot_score: Tensor,
        temp_sampling: float,
        temp_psi: float,
        temp_sigma_data_tr: float,
        temp_sigma_data_rot: float,
    ):
        tr_sigma_data = np.exp(
            temp_sigma_data_tr * np.log(self.noise_transform.noise_schedule.tr_s_max)
            + (1 - temp_sigma_data_tr)
            * np.log(self.noise_transform.noise_schedule.tr_s_min)
        )
        lambda_tr = (tr_sigma_data + tr_s) / (tr_sigma_data + tr_s / temp_sampling)
        tr_update = (
            tr_g**2 * dt * (lambda_tr + temp_sampling * temp_psi / 2) * tr_score.cpu()
            + tr_g * np.sqrt(dt * (1 + temp_psi)) * tr_z
        ).cpu()

        rot_sigma_data = np.exp(
            temp_sigma_data_rot * np.log(self.noise_transform.noise_schedule.rot_s_max)
            + (1 - temp_sigma_data_rot)
            * np.log(self.noise_transform.noise_schedule.rot_s_min)
        )
        lambda_rot = (rot_sigma_data + rot_s) / (rot_sigma_data + rot_s / temp_sampling)
        rot_update = (
            rot_g**2
            * dt
            * (lambda_rot + temp_sampling * temp_psi / 2)
            * rot_score.cpu()
            + rot_g * np.sqrt(dt * (1 + temp_psi)) * rot_z
        ).cpu()

        return tr_update, rot_update
