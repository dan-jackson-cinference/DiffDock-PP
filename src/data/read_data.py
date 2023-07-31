from __future__ import annotations
import os
import pickle
from abc import ABC, abstractmethod

import dill
import pandas as pd
import torch
from tqdm import tqdm

from config import DataCfg
from data.protein import PPComplex, parse_pdb
from utils import load_csv

# -------- DATA LOADING -------
DATA_CACHE_VERSION = "v2"

PPDataSet = dict[str, PPComplex]


class DataReader(ABC):
    def __init__(
        self,
        experiment_root_dir: str,
        data_root_dir: str,
        data_file: str,
        recache: bool,
        debug: bool,
    ):
        self.experiment_root_dir = experiment_root_dir
        self.data_root_dir = data_root_dir
        self.data_file = data_file
        self.recache = recache
        self.debug = debug

        self.data_cache = self.data_file.replace(
            ".csv", f"_cache_{DATA_CACHE_VERSION}.pkl"
        )

    @abstractmethod
    def read_files(self, data_to_load: list[dict[str, str]]) -> PPDataSet:
        raise NotImplementedError

    @classmethod
    def from_config(cls, experiment_root_dir: str, cfg: DataCfg) -> DataReader:
        return cls(
            experiment_root_dir,
            cfg.structures_dir,
            cfg.data_file,
            cfg.recache,
            cfg.debug,
        )

    def load_data(self):
        data_to_load = load_csv(os.path.join(self.experiment_root_dir, self.data_file))
        return self.read_files(data_to_load)

    def _to_dict(self, pdb_id: str, p1, p2) -> dict[str, str]:
        item = {"pdb_id": pdb_id, "receptor": p1, "ligand": p2}
        return item


class DIPSReader(DataReader):
    def __init__(self, data_root_dir: str, data_file: str, recache: bool, debug: bool):
        super().__init__(data_root_dir, data_file, recache, debug)

    def read_files(self, data_to_load: list[dict[str, str]]):
        data = {}
        # check if loaded previously
        if not self.recache and os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}
        for line in tqdm(data_to_load, desc="data loading", ncols=50):
            path = line["path"]
            item = path_to_data.get(path)
            if item is None:
                item = self.parse_dill(os.path.join(self.data_root_dir, path), path)
                path_to_data[path] = item
            item.update(line)  # add meta-data
            data[path] = item
            if self.debug:
                if len(data) >= 2:
                    break
        # write to cache
        if self.recache or not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return data

    def parse_dill(self, fp, pdb_id):
        with open(fp, "rb") as f:
            data = dill.load(f)
        p1, p2 = data[1], data[2]
        p1, p2 = self.parse_df(p1), self.parse_df(p2)
        return self._to_dict(pdb_id, p1, p2)

    def parse_df(self, df: pd.DataFrame):
        """
        Parse PDB DataFrame
        """
        # extract dataframe values
        all_res = df["resname"]
        all_pos = torch.tensor([df["x"], df["y"], df["z"]]).t()
        all_atom = list(zip(df["atom_name"], df["element"]))
        visualization_values = {
            "chain": df["chain"],
            "resname": all_res,
            "residue": df["residue"],
            "atom_name": df["atom_name"],
            "element": df["element"],
        }
        # convert to seq, pos
        return all_res, all_atom, all_pos, visualization_values


class DB5Reader(DataReader):
    def __init__(
        self,
        data_root_dir: str,
        data_file: str,
        recache: bool,
        debug: bool,
        use_unbound: bool,
    ):
        super().__init__(data_root_dir, data_file, recache, debug)
        self.use_unbound = use_unbound

    @classmethod
    def from_config(cls, cfg: DataCfg) -> DataReader:
        return cls(
            cfg.structures_dir, cfg.data_file, cfg.recache, cfg.debug, cfg.use_unbound
        )

    def read_files(self, data_to_load: list[dict[str, str]]) -> PPDataSet:
        ppi_data: PPDataSet = {}
        # check if loaded previously
        if not self.recache and os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}

        for line in tqdm(data_to_load, desc="data loading", ncols=50):
            pdb_id = line["path"]
            split = line["split"]
            # item = path_to_data.get(pdb_id)
            # if item is None:
            pp_complex = self.parse_path(
                pdb_id, os.path.join(self.data_root_dir, pdb_id), split
            )
            # path_to_data[pdb_id] = pp_complex
            ppi_data[pdb_id] = pp_complex
            if self.debug:
                if len(ppi_data) >= 2:
                    break
        # write to cache
        if self.recache or not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return ppi_data

    def parse_path(self, pdb_id: str, path: str, split: str) -> PPComplex:
        if self.use_unbound:
            fp_rec = f"{path}_r_u.pdb"
            fp_lig = f"{path}_l_u.pdb"
        else:
            fp_rec = f"{path}_r_b.pdb"
            fp_lig = f"{path}_l_b.pdb"
        receptor = parse_pdb(fp_rec, pdb_id)
        ligand = parse_pdb(fp_lig, pdb_id)
        return PPComplex(pdb_id, receptor, ligand, split)


class CSVReader(DataReader):
    def __init__(
        self,
        experiment_root_dir: str,
        data_root_dir: str,
        data_file: str,
        recache: bool,
        debug: bool,
        use_unbound: bool,
    ):
        super().__init__(experiment_root_dir, data_root_dir, data_file, recache, debug)
        self.use_unbound = use_unbound

    @classmethod
    def from_config(cls, experiment_root_dir: str, cfg: DataCfg) -> DataReader:
        return cls(
            experiment_root_dir,
            cfg.structures_dir,
            cfg.data_file,
            cfg.recache,
            cfg.debug,
            cfg.use_unbound,
        )

    def read_files(self, data_to_load: list[dict[str, str]]) -> PPDataSet:
        ppi_data: PPDataSet = {}
        # check if loaded previously
        if not self.recache and os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}

        for line in tqdm(data_to_load, desc="data loading", ncols=50):
            pdb_id = line["pdb_id"]
            # item = path_to_data.get(pdb_id)
            # if item is None:
            receptor, ligand = parse_pdb(
                f"{os.path.join(self.experiment_root_dir, self.data_root_dir, pdb_id)}.pdb",
                pdb_id,
                line["receptor"],
                line["ligand"],
            )
            pp_complex = PPComplex(pdb_id, receptor, ligand, "test")
            # path_to_data[pdb_id] = pp_complex
            ppi_data[pdb_id] = pp_complex
            if self.debug:
                if len(ppi_data) >= 2:
                    break
        # write to cache
        if self.recache or not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return ppi_data


class SinglePairReader(DataReader):
    def __init__(
        self,
        pdb_id: str,
        pdb_path: str,
        receptor_chains: list[str],
        ligand_chains: list[str],
    ):
        self.pdb_id = pdb_id
        self.pdb_path = pdb_path
        self.receptor_chains = receptor_chains
        self.ligand_chains = ligand_chains

    @classmethod
    def from_config(cls, cfg: DataCfg) -> DataReader:
        pred_cfg = cfg.pred_cfg
        return cls(
            pred_cfg.pdb_id,
            pred_cfg.pdb_path,
            pred_cfg.receptor_chains,
            pred_cfg.ligand_chains,
        )

    def load_data(self):
        return self.read_files()

    def read_files(self) -> PPDataSet:
        receptor, ligand = parse_pdb(
            self.pdb_path, self.pdb_id, self.receptor_chains, self.ligand_chains
        )
        return {self.pdb_id: PPComplex(self.pdb_id, receptor, ligand, "test")}
