from copy import deepcopy
import copy
from typing import Optional
import torch

from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Dataset, HeteroData

from data.protein import PPComplex
from geom_utils.transform import NoiseTransform

# ------ DATASET -------


class BindingDataset(Dataset):
    """
    Protein-protein binding dataset
    """

    def __init__(
        self,
        data: dict[str, PPComplex],
        noise_transform: Optional[NoiseTransform] = None,
        pdb_ids: Optional[list[str]] = None,
    ):
        super().__init__(transform=noise_transform)
        # select subset for given split
        if pdb_ids is not None:
            data = {
                pdb_id: pp_complex
                for pdb_id, pp_complex in data.items()
                if pdb_id in pdb_ids
            }
            self.pdb_ids = [k for k in data if k in pdb_ids]
        else:
            self.pdb_ids = list(data)
        # convert to PyTorch geometric objects upon GET not INIT
        self.data_list = list(data.values())

    def len(self):
        return len(self.data_list)

    def __delitem__(self, idx: int) -> None:
        """
        Easier deletion interface. MUST update length.
        """
        del self.data_list[idx]

    def get(self, idx: int) -> HeteroData:
        """
        Create graph object to keep original object intact,
        so we can modify pos, etc.
        """
        return deepcopy(self.data_list[idx].graph)

    def get_pp_complex(
        self, pdb_name: Optional[str] = None, index: Optional[int] = None
    ) -> PPComplex:
        if index is not None:
            pass
        elif pdb_name is not None:
            index = self.pdb_ids.index(pdb_name)
        else:
            raise Exception("Either pdb_name or index should be given!")

        return self.data_list[index]

    def set_graph(self, idx: int, new_graph: HeteroData):
        self.data_list[idx].graph = new_graph


class RandomizedConfidenceDataset(Dataset):
    """
    Protein-protein dataset of randomly perturbed ligand poses used to train a confidence model. (experimental)
    """

    def __init__(self, data, pdb_ids=None):
        super(RandomizedConfidenceDataset, self).__init__()
        # select subset for given split
        if pdb_ids is not None:
            data = {k: v for k, v in data.items() if k in pdb_ids}
            self.pdb_ids = [k for k in data if k in pdb_ids]
        else:
            self.pdb_ids = list(data)
        # convert to PyTorch geometric objects upon GET not INIT
        self.data = list(data.values())
        self.length = len(self.data)

    def len(self):
        return self.length

    def __delitem__(self, idx):
        """
        Easier deletion interface. MUST update length.
        """
        del self.data[idx]
        self.len = len(self.data)

    def get(self, idx):
        """
        Create graph object to keep original object intact,
        so we can modify pos, etc.
        """
        item = self.data[idx]["graph"]
        # >>> fix this later no need to copy tensors only references
        data = copy.deepcopy(item)
        set_time(data, 0, 0, 0, 1)
        if np.random.rand() < 0.05:
            tr_s_max = 5
            rot_s_max = 0.2
        else:
            tr_s_max = 2
            rot_s_max = 0.1
        return self.randomize_position_and_compute_rmsd(
            data, tr_s_max=tr_s_max, rot_s_max=rot_s_max
        )  # 2 and 0.1 yields almost 50% -> good. 2 and 0.2 yields 13%

    def set_graph(self, idx, new_graph):
        self.data[idx]["graph"] = new_graph

    def randomize_position_and_compute_rmsd(self, complex_graph, tr_s_max, rot_s_max):
        # randomize rotation
        original_pos = complex_graph["ligand"].pos
        center = torch.mean(original_pos, dim=0, keepdim=True)
        # one way to generate random rotation matrix
        # random_rotation = torch.from_numpy(R.random().as_matrix())

        # Another way
        rot_update = sample_vec(eps=rot_s_max)  # * rot_s_max
        rot_update = torch.from_numpy(rot_update).float()
        random_rotation = axis_angle_to_matrix(rot_update.squeeze())

        # yet another way
        # x = np.random.randn(3)
        # x /= np.linalg.norm(x)
        # x *= rot_s_max
        # random_rotation = R.from_euler('zyx', x, degrees=True)
        # random_rotation = torch.from_numpy(random_rotation.as_matrix())
        pos = (original_pos - center) @ random_rotation.T.float()

        # random translation
        tr_update = torch.normal(0, tr_s_max, size=(1, 3))
        pos = pos + tr_update + center
        complex_graph["ligand"].pos = pos

        # compute rmsd
        rmsd = compute_rmsd(original_pos, pos)
        return complex_graph, rmsd


class SamplingDataset(Dataset):
    """
    Protein-protein binding dataset
    """

    def __init__(
        self,
        pp_complex: PPComplex,
        num_samples: int,
        tr_s_max: float,
        noise_transform: Optional[NoiseTransform] = None,
    ):
        super().__init__(transform=noise_transform)
        # select subset for given split
        self.data = initialize_random_positions(
            pp_complex.graph,
            num_samples,
            tr_s_max,
            no_torsion=True,
        )

    def len(self):
        return len(self.data)

    def __delitem__(self, idx: int) -> None:
        """
        Easier deletion interface. MUST update length.
        """
        del self.data[idx]

    def get(self, idx: int) -> HeteroData:
        """
        Create graph object to keep original object intact,
        so we can modify pos, etc.
        """
        return self.data[idx]


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
