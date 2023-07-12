import logging

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore

from data.load_data import load_data
from src.config import DiffDockCfg

cs = ConfigStore.instance()
cs.store(name="test_config", node=DiffDockCfg)


@hydra.main(version_base=None, config_path="configs", config_name="diffdock")
def main(cfg: DiffDockCfg):
    """main function"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    np.random.seed(0)

    data = load_data(cfg.data_conf)
