import os

import torch
from torch import nn
from torch.optim import Adam

from utils import printt


def get_optimizer(
    model: nn.Module,
    weight_decay: float,
    lr: float,
    load_best=True,
    confidence_mode=False,
):
    """
    Initialize optimizer and load if applicable
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = SGD(model.parameters(),
    #                lr=args.lr,
    #                momentum=0.5,
    #                weight_decay=args.weight_decay)
    # SGD is awful for my models. don't use it.
    ## load optimizer state
    fold_dir = args.fold_dir
    if args.checkpoint_path is not None:
        if load_best:
            checkpoint = select_model(fold_dir, confidence_mode=confidence_mode)
        else:
            checkpoint = os.path.join(fold_dir, "model_last.pth")

        if checkpoint is not None:
            # start_epoch = int(checkpoint.split("/")[-1].split("_")[3])
            start_epoch = 0
            with torch.no_grad():
                optimizer.load_state_dict(
                    torch.load(checkpoint, map_location="cpu")["optimizer"]
                )
            printt("Finished loading optimizer")
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    return start_epoch, optimizer
