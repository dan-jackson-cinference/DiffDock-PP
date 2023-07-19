import copy
from typing import Optional, Type, Any

import torch
from torch_geometric.data import Dataset

# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataListLoader, DataLoader

from data.dataset import BindingDataset, RandomizedConfidenceDataset
from data.process_data import (
    crossval_split,
    split_data,
    split_into_folds,
)
from data.protein import PPComplex
from geom_utils.transform import NoiseTransform


def split_data(
    ppi_data: dict[str, PPComplex],
    fold_num: int,
    num_folds: int,
    mode: str = "test",
    debug: bool = False,
) -> dict[str, dict[str, PPComplex]]:
    """
    Convert raw data into DataLoaders for training.
    """

    if debug:  # and False: TODO
        splits = {}
        splits["train"] = dataset_class(
            args, dataset.data, apply_transform=not for_reverse_diffusion
        )
        splits["train"].data = [
            splits["train"].data[1]
        ] * args.multiplicity  # take first element and repeat it
        splits["train"].length = len(splits["train"].data)
        splits["val"] = dataset_class(
            args, dataset.data, apply_transform=not for_reverse_diffusion
        )
        splits["val"].data = [splits["val"].data[1]]  # take first element
        splits["val"].length = len(splits["val"].data)
        splits["test"] = copy.deepcopy(splits["train"])  # TODO
        print("test", len(splits["test"]), len(splits["train"]))
        if for_reverse_diffusion:
            return splits
        return _get_loader(splits, args)
    # use_pose = isinstance(dataset_class, tuple)
    # if use_pose:
    #     dataset, poses = dataset
    # smush folds and convert to Dataset object
    # or extract val and rest are train
    # for training, without crossval_split, weird stuff happens
    splits = split_data(ppi_data, fold_num, num_folds)
    # split train into separate folds
    # sometimes we only want to load val data. Then train data is empty
    if len(splits["train"]) > 0:
        splits["train"] = split_into_folds(splits["train"], num_folds)

    if mode == "train":
        splits = crossval_split(splits, fold_num, num_folds)

    return splits


def get_datasets(
    splits: dict[str, dict[str, PPComplex]],
    noise_transform: Optional[NoiseTransform],
    dataset_class: Type[Dataset],
) -> dict[str, Dataset]:
    dataset_splits = {}
    for split, pdb_ids in splits.items():
        dataset_splits[split] = dataset_class(split, noise_transform, pdb_ids)
    return dataset_splits


def get_loaders(
    splits: dict[str, Dataset],
    batch_size: int,
    num_workers: int,
    num_gpu: int,
    mode: str = "test",
) -> dict[str, DataLoader]:
    """
    Convert lists into DataLoader
    """
    # current reverse diffusion does NOT use DataLoader
    if mode == "test":
        return splits
    # convert to DataLoader
    loaders: dict[str, DataLoader] = {}
    for split, data in splits.items():
        # account for test-only datasets
        if len(data) == 0:
            loaders[split] = []
            continue
        # do not shuffle val/test
        shuffle = split == "train"
        # set proper DataLoader object (PyG)
        if torch.cuda.is_available() and num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader
        loaders[split] = loader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=shuffle,
        )
    return loaders


def generate_loaders(
    loader, args
) -> tuple[list[BindingDataset], list[list[tuple[Any, float]]]]:
    result: list[BindingDataset] = []
    data = loader.data
    ground_truth: list[list[tuple[Any, float]]] = []
    for d in data:
        # element = BindingDataset(args, {}, apply_transform=False)
        data_list = []

        for i in range(args.num_samples):
            data_list.append(copy.deepcopy(d))

        if args.mirror_ligand:
            # printt('Mirroring half of the complexes')
            for i in range(0, args.num_samples // 2):
                e = data_list[i]["graph"]

                data = HeteroData()
                data["name"] = e["name"]

                data["receptor"].pos = e["ligand"].pos
                data["receptor"].x = e["ligand"].x

                data["ligand"].pos = e["receptor"].pos
                data["ligand"].x = e["receptor"].x

                data["receptor", "contact", "receptor"].edge_index = e[
                    "ligand", "contact", "ligand"
                ].edge_index
                data["ligand", "contact", "ligand"].edge_index = e[
                    "receptor", "contact", "receptor"
                ].edge_index

                # center receptor at origin
                center = data["receptor"].pos.mean(dim=0, keepdim=True)
                for key in ["receptor", "ligand"]:
                    data[key].pos = data[key].pos - center
                data.center = center  # save old center
                data["mirrored"] = True

                data_list[i]["graph"] = data

        element = BindingDataset(args, data_list, apply_transform=False)
        # element.data = data_list
        # element.length = args.num_samples
        # print(f'2: {element[2]}')
        # print(f'2_ligand: {element[2]["ligand"]}')
        # print(f'2_ligand_num_nodes: {element[2]["ligand"].num_nodes}')
        # print(f'0: {element[0]["ligand"].num_nodes}')

        result.append(element)

    for element in loader:
        ground_truth.append([(element, float("inf"))])
    return result, ground_truth
