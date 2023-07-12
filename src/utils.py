import csv
import glob
import itertools
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

# -------- general


def load_csv(
    csv_path: str, split: Optional[str] = None, batch: Optional[str] = None
) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            print(line)
            # Skip not specified batches & splits
            if split is not None and line["split"] != split:
                continue
            if batch is not None and int(line["batch"]) != int(batch):
                continue
            data.append(line)
    return data


def log(item, fp, reduction=True):
    # pre-process item
    item_new = {}
    for key, val in item.items():
        if type(val) is list and reduction:
            key_std = f"{key}_std"
            item_new[key] = float(np.mean(val))
            item_new[key_std] = float(np.std(val))
        else:
            if torch.is_tensor(val):
                item[key] = val.tolist()
            item_new[key] = val
    # initialization: write keys
    if not os.path.exists(fp):
        with open(fp, "w+") as f:
            f.write("")
    # append values
    with open(fp, "a") as f:
        yaml.dump(item_new, f)
        f.write(os.linesep)


def chain(iterable, as_set=True):
    if as_set:
        return sorted(set(itertools.chain.from_iterable(iterable)))
    else:
        return list(itertools.chain.from_iterable(iterable))


def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")


def get_unixtime():
    timestamp = str(int(time.time()))
    return timestamp


def printt(*args, **kwargs) -> None:
    print(get_timestamp(), *args, **kwargs)


def print_res(scores):
    """
    @param (dict) scores key -> score(s)
    """
    for key, val in scores.items():
        if type(val) is list:
            print_str = f"{np.mean(val):.3f} +/- {np.std(val):.3f}"
            print_str = print_str + f" ({len(val)})"
        else:
            print_str = f"{val:.3f}"
        print(f"{key}\t{print_str}")


def get_model_path(fold_dir) -> str:
    # load last model saved (we only save if improvement in validation performance)
    # convoluted code says "sort by epoch, then batch"
    # new code says "sort by rmsd, take the lowest"
    paths = []
    for path in glob.glob(f"{fold_dir}/*.pth"):
        if "last" not in path:
            paths.append(path)
    models = sorted(paths, key=lambda s: float(s.split("/")[-1].split("_")[4]))
    # key=lambda s:(int(s.split("/")[-1].split("_")[3]),
    #              int(s.split("/")[-1].split("_")[2])))
    if len(models) == 0:
        print(f"no models found at {fold_dir}")
        return
    checkpoint = models[0]
    return checkpoint


def select_model(fold_dir, confidence_mode):
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


def init(model):
    """
    Wrapper around Xavier normal initialization
    Apparently this needs to be called in __init__ lol
    """
    for name, param in model.named_parameters():
        # NOTE must name parameter "bert"
        if "bert" in name:
            continue
        # bias terms
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        # weight terms
        else:
            nn.init.xavier_normal_(param)


if __name__ == "__main__":
    test = load_csv(
        "/Users/danieljackson/Projects/Cinference/DiffDock-PP/datasets/single_pair_dataset/splits_test.csv"
    )
    print(test)
