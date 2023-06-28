from abc import ABC, abstractmethod
from typing import Optional

import wandb
from torch.utils.tensorboard import SummaryWriter


class Logger(ABC):
    """Abstract class for logger."""

    @abstractmethod
    def add_scalar(
        self, tag: str, scalar_value: float, global_step: Optional[int] = None
    ) -> None:
        """Logs scalar value."""
        raise NotImplementedError


class WandbLogger(Logger):
    """Weight and Biases logger."""

    def __init__(
        self,
        project: str,
        entity: str,
        name: str = None,
        group: str = None,
        config=None,
        notes: str = None,
    ):
        """
        notes: A longer description of the run. helps you remember what you were doing when you
            ran this run.
        """
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            config=config,
            notes=notes,
        )

    @classmethod
    def construct_logger(
        cls, project: str, entity: str, name: str, group: str, config: str, notes: str
    ):
        return cls(project, entity, name, group, config, notes)

    def add_scalar(self, tag: str, scalar_value, global_step=None):
        wandb.log({tag: scalar_value}, step=global_step)


class TensorboardLogger(Logger):
    """Tensorboard logger."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    @classmethod
    def construct_logger(cls, log_dir: str):
        return cls(log_dir)

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: Optional[int] = None
    ):
        self.writer.add_scalar(tag, scalar_value, global_step)


LOGGERS = {"tensorboard": TensorboardLogger, "wandb": WandbLogger}


def constuct_logger(logger_conf) -> Logger:
    return LOGGERS[logger_conf.name].construct_logger(logger_conf)
