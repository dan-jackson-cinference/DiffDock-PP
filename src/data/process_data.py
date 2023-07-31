from __future__ import annotations
from abc import ABC, abstractmethod
import os
import pickle
import random
import warnings
from typing import Any


import Bio
import Bio.PDB
import torch
from torch import Tensor
from tqdm import tqdm
from config import DataCfg

from data.protein import PPComplex, Protein
from data.tokenise import tokenize

warnings.filterwarnings("ignore", category=Bio.PDB.PDBExceptions.PDBConstructionWarning)

from utils import printt

DATA_CACHE_VERSION = "v2"

PPDataSet = dict[str, PPComplex]


ATOMS_TO_KEEP = {"atom": None, "backbone": ["N", "CA", "O", "CB"], "residue": ["CA"]}


class DataProcessor(ABC):
    """
    Base preprocessing class
    @param (bool) data  default=True to load complexes, not poses
    """

    def __init__(
        self,
        root_dir: str,
        data_file: str,
        resolution: str,
        no_graph_cache: bool,
        use_orientation_features: bool,
        recache: bool,
        knn_size: int,
        lm_embed_dim: int,
        debug: bool = False,
        cache_dir: str = "cache",
    ):
        self.resolution = resolution
        self.no_graph_cache = no_graph_cache
        self.use_orientation_features = use_orientation_features
        self.debug = debug
        self.recache = recache
        self.lm_embed_dim = lm_embed_dim
        self.knn_size = knn_size
        self.cache_dir = os.path.join(root_dir, cache_dir)
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        # cache for post-processed data, optional
        self.graph_cache = data_file.replace(".csv", f"_graph_{self.resolution}.pkl")
        self.esm_cache = data_file.replace(".csv", f"_esm.pkl")
        # if self.use_orientation_features:
        #    self.data_cache = self.data_cache.replace(".pkl", "_orientation.pkl")
        #    self.graph_cache = self.graph_cache.replace(".pkl", "_orientation.pkl")
        #    self.esm_cache = self.esm_cache.replace(".pkl", "_orientation.pkl")
        if debug:
            # self.data_cache = data_cache.replace(".pkl", "_debug.pkl")
            self.graph_cache = self.graph_cache.replace(".pkl", "_debug.pkl")
            self.esm_cache = self.esm_cache.replace(".pkl", "_debug.pkl")
        # biopython PDB parser
        self.parser = Bio.PDB.PDBParser()

        # load pretrained model
        self.esm_model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", "esm2_t33_650M_UR50D"
        )
        self.esm_model.cuda().eval()
        self.tokenizer = self.alphabet.get_batch_converter()

    @abstractmethod
    def process(self, data: PPDataSet) -> tuple[PPDataSet, dict[str, int]]:
        raise NotImplementedError

    def extract_coords(self, protein: Protein) -> dict[str, list[Tensor]]:
        """
        Extract coordinates of CA, N, and C atoms that are needed
        to compute orientation features, as proposed in EquiDock

        @return c_alpha_coords, n_coords, c_coords (all torch.Tensor)
        """
        coords: dict[str, list[Tensor]] = {"CA": [], "N": [], "C": []}
        for atom in protein.sequence:
            if atom.name in coords:
                coords[atom.name].append(atom.pos)

        assert len(coords["CA"]) == len(coords["N"]) == len(coords["C"])

        return coords

    def preprocess_data(
        self,
        ppi_data: PPDataSet,
    ) -> tuple[PPDataSet, dict[str, Any]]:
        """
        Tokenize, etc.
        """
        # check if cache exists

        if not self.no_graph_cache and os.path.exists(self.graph_cache):
            with open(self.graph_cache, "rb") as f:
                ppi_data, data_params = pickle.load(f)
                printt("Loaded processed data from cache")
                return ppi_data, data_params

        data_params: dict[str, Any] = {}

        for pp_complex in ppi_data.values():
            for protein in [pp_complex.receptor, pp_complex.ligand]:
                # select subset of residues that match desired resolution (e.g. residue-level vs. atom-level)
                protein.filter(ATOMS_TO_KEEP[self.resolution])
                # for orientation features we need coordinates of CA, N, and C atoms
                # These coordinates define the local coordinate system of each residue
                if self.use_orientation_features:
                    coords = self.extract_coords(protein)
                    protein.compute_orientation_vectors(
                        coords["CA"], coords["N"], coords["C"]
                    )

        #### convert to HeteroData graph objects
        for pp_complex in ppi_data.values():
            pp_complex.populate_graph(self.use_orientation_features, self.knn_size)

        if not self.no_graph_cache:
            with open(self.graph_cache, "wb+") as f:
                pickle.dump(ppi_data, f)

        return ppi_data, data_params

    def process_embed(
        self,
        ppi_data: PPDataSet,
        data_params: dict[str, Any],
    ):
        #### tokenize AFTER converting to graph
        if self.lm_embed_dim > 0:
            ppi_data = self.compute_embeddings(ppi_data)
            data_params["num_residues"] = 23  # <cls> <sep> <pad>
            printt("finished tokenizing residues with ESM")
        else:
            # tokenize residues for non-ESM
            tokenizer = tokenize(ppi_data.values())
            tokenize(ppi_data.values(), tokenizer)
            self.esm_model = None
            data_params["num_residues"] = len(tokenizer)
            data_params["tokenizer"] = tokenizer
            printt("finished tokenizing residues")

        #### protein sequence tokenization
        # tokenize atoms
        atom_tokenizer = tokenize(
            [pp_complex.receptor for pp_complex in ppi_data.values()]
        )
        tokenize(
            [pp_complex.ligand for pp_complex in ppi_data.values()], atom_tokenizer
        )
        data_params["atom_tokenizer"] = atom_tokenizer
        printt("finished tokenizing all inputs")

        return ppi_data, data_params

    def compute_embeddings(self, ppi_data: PPDataSet):
        """
        Pre-compute ESM2 embeddings.
        """
        # check if we already computed embeddings
        print(os.path.exists(self.esm_cache))
        if os.path.exists(self.esm_cache):
            with open(self.esm_cache, "rb") as file:
                path_to_rep = pickle.load(file)
            self._save_esm_rep(ppi_data, path_to_rep)
            printt("Loaded cached ESM embeddings")
            return ppi_data

        printt("Computing ESM embeddings")
        # fix ordering
        all_pdbs = sorted(ppi_data)

        rec_seqs: list[str] = []
        lig_seqs: list[str] = []
        for pp_complex in ppi_data.values():
            rec_seqs.append(pp_complex.receptor.filtered_sequence_str)
            lig_seqs.append(pp_complex.ligand.filtered_sequence_str)

        # batchify sequences
        rec_batches = self._esm_batchify(rec_seqs)
        lig_batches = self._esm_batchify(lig_seqs)
        with torch.no_grad():
            rec_reps = self._run_esm(rec_batches)
            lig_reps = self._run_esm(lig_batches)

        # dump to cache
        path_to_rep: dict[str, tuple[Tensor, Tensor]] = {}

        for idx, pdb in enumerate(all_pdbs):
            # cat one-hot representation and ESM embedding
            rec_graph_x = torch.cat([rec_reps[idx][0], rec_reps[idx][1]], dim=1)
            lig_graph_x = torch.cat([lig_reps[idx][0], lig_reps[idx][1]], dim=1)
            path_to_rep[pdb] = rec_graph_x, lig_graph_x
        with open(self.esm_cache, "wb+") as file:
            pickle.dump(path_to_rep, file)

        # overwrite graph.x for each element in batch
        self._save_esm_rep(ppi_data, path_to_rep)

        return ppi_data

    def _esm_batchify(self, seqs: list[str]) -> list[Tensor]:
        batch_size = find_largest_diviser_smaller_than(len(seqs), 32)

        iterator = range(0, len(seqs), batch_size)
        # group up sequences
        batches = [seqs[i : i + batch_size] for i in iterator]
        batches = [[("", seq) for seq in batch] for batch in batches]
        # tokenize
        batch_tokens = [self.tokenizer(batch)[2] for batch in batches]
        return batch_tokens

    def _run_esm(self, batches: list[Tensor]) -> list[tuple[Tensor, Tensor]]:
        """
        Wrapper around ESM specifics
        @param (list)  batch
        @return (list)  same order as batch
        """
        # run ESM model
        all_reps: list[Tensor] = []
        for batch in tqdm(batches, desc="ESM", ncols=50):
            reps = self.esm_model(batch.cuda(), repr_layers=[33])
            # reps = reps["representations"][33].cpu().squeeze()[:,1:]
            # remove squeeze() to allow for running on a single pair.
            reps = reps["representations"][33].cpu()[:, 1:]
            all_reps.append(reps)
        # crop to length
        # exclude <cls> <sep>
        cropped: list[tuple[Tensor, Tensor]] = []
        for i, batch in enumerate(batches):
            batch_lens = (batch != self.alphabet.padding_idx).sum(1) - 2
            for j, length in enumerate(batch_lens):
                rep_crop = all_reps[i][j, :length]
                token_crop = batch[j, 1 : length + 1, None]
                cropped.append((rep_crop, token_crop))
        return cropped

    def _save_esm_rep(
        self,
        ppi_data: PPDataSet,
        path_to_rep: dict[str, tuple[Tensor, Tensor]],
    ) -> PPDataSet:
        """
        Assign new ESM representation to graph.x
        """
        for pdb, (rec_rep, lig_rep) in path_to_rep.items():
            rec_graph = ppi_data[pdb].graph["receptor"]
            lig_graph = ppi_data[pdb].graph["ligand"]
            rec_graph.x = rec_rep
            lig_graph.x = lig_rep

            assert len(rec_graph.pos) == len(rec_graph.x)
            assert len(lig_graph.pos) == len(lig_graph.x)
        return ppi_data

    @classmethod
    def from_config(
        cls, root_dir: str, cfg: DataCfg, lm_embed_dim: int, debug: bool
    ) -> DataProcessor:
        return cls(
            root_dir,
            cfg.data_file,
            cfg.resolution,
            cfg.no_graph_cache,
            cfg.use_orientation_features,
            cfg.recache,
            cfg.knn_size,
            lm_embed_dim,
            debug,
        )


class DIPSProcessor(DataProcessor):
    def __init__(
        self,
        data_file: str,
        resolution: str,
        no_graph_cache: bool,
        use_orientation_features: bool,
        recache: bool,
        knn_size: int,
        lm_embed_dim: int,
        debug: bool,
    ):
        super().__init__(
            data_file,
            resolution,
            no_graph_cache,
            use_orientation_features,
            recache,
            knn_size,
            lm_embed_dim,
            debug,
        )

        self.lm_embed_dim = lm_embed_dim

    def assign_receptor(self, data: PPDataSet):
        """
        For docking, we assigned smaller protein as ligand
        for this dataset (since no canonical receptor/ligand
        assignments)
        """
        for item in data.values():
            rec = item["receptor"]
            lig = item["ligand"]
            if len(rec[0]) < len(lig[0]):
                item["receptor"] = lig
                item["ligand"] = rec
        return data

    def process(self, data: PPDataSet):
        data = self.assign_receptor(data)
        data, data_params = self.preprocess_data(data)
        data, data_params = self.process_embed(data, data_params)
        #### pre-compute ESM embeddings if needed
        printt(len(data), "entries loaded")
        return data, data_params


class DB5Processor(DataProcessor):
    """
    Docking benchmark 5.5
    """

    def __init__(
        self,
        data_file: str,
        resolution: str,
        no_graph_cache: bool,
        use_orientation_features: bool,
        recache: bool,
        knn_size: int,
        lm_embed_dim: int,
        debug: bool,
        use_unbound: bool,
    ):
        super().__init__(
            data_file,
            resolution,
            no_graph_cache,
            use_orientation_features,
            recache,
            knn_size,
            lm_embed_dim,
            debug,
        )
        # cache file dependent on use_unbound
        if use_unbound:
            printt("Using Unbound structures")
            # self.data_cache = self.data_cache.replace(".pkl", "_u.pkl")
            self.esm_cache = self.esm_cache.replace(".pkl", "_u.pkl")
        else:
            printt("Using Bound structures")
            # self.data_cache = self.data_cache.replace(".pkl", "_b.pkl")
            self.esm_cache = self.esm_cache.replace(".pkl", "_b.pkl")

    def process(self, data: PPDataSet):
        ppi_data, data_params = self.preprocess_data(data)
        ppi_data, data_params = self.process_embed(ppi_data, data_params)
        printt(len(ppi_data), "entries loaded")
        return ppi_data, data_params


class SabDabProcessor(DataProcessor):
    """
    Structure Antibody Database
    Downloaded May 2, 2022.
    """

    def __init__(
        self,
        data_file: str,
        resolution: str,
        no_graph_cache: bool,
        use_orientation_features: bool,
        recache: bool,
        knn_size: int,
        lm_embed_dim: int,
        debug: bool,
    ):
        super().__init__(
            data_file,
            resolution,
            no_graph_cache,
            use_orientation_features,
            recache,
            knn_size,
            lm_embed_dim,
            debug,
        )

    def process(self, data: PPDataSet):
        data, data_params = self.preprocess_data(data)
        data, data_params = self.process_embed(data, data_params)
        #### pre-compute ESM embeddings if needed
        printt(len(data), "entries loaded")
        return data, data_params


class SinglePairProcessor(DataProcessor):
    def __init__(
        self,
        root_dir: str,
        data_file: str,
        resolution: str,
        no_graph_cache: bool,
        use_orientation_features: bool,
        recache: bool,
        knn_size: int,
        lm_embed_dim: int,
        debug: bool,
    ):
        super().__init__(
            root_dir,
            data_file,
            resolution,
            no_graph_cache,
            use_orientation_features,
            recache,
            knn_size,
            lm_embed_dim,
            debug,
        )
        self.data_cache = os.path.join(
            self.cache_dir, f"test_data_cache_{DATA_CACHE_VERSION}.pkl"
        )
        self.esm_cache = os.path.join(self.cache_dir, "test_esm.pkl")
        self.graph_cache = os.path.join(self.cache_dir, f"test_graph_{resolution}.pkl")

    def process(self, data: PPDataSet):
        ppi_data, data_params = self.preprocess_data(data)
        ppi_data, data_params = self.process_embed(ppi_data, data_params)
        printt(len(ppi_data), "entries loaded")
        return ppi_data, data_params


# ------ DATA PROCESSING ------
def split_into_folds(ppi_data: PPDataSet, num_folds: int) -> list[PPDataSet]:
    """
    Split into train/val/test folds
    @param (list) data
    """
    keys = sorted(ppi_data.keys())
    random.shuffle(keys)
    # split into folds
    cur_clst = 0
    folds: list[PPDataSet] = [{} for _ in range(num_folds)]
    fold_ratio = 1.0 / num_folds
    max_fold_size = int(len(ppi_data) * fold_ratio)
    for fold_num in range(num_folds):
        fold: list[str] = []
        while len(fold) < max_fold_size:
            fold.append(keys[cur_clst])
            cur_clst += 1
        for k in fold:
            folds[fold_num][k] = ppi_data[k]
    return folds


# ------ DATA COLLATION -------


def split_data(raw_data: PPDataSet, num_folds: int) -> dict[str, PPDataSet]:
    """separate out train/test"""

    data: dict[str, PPDataSet] = {"train": {}, "val": {}, "test": {}}
    ## if test split is pre-specified, split into train/test
    # otherwise, allocate all data to train for cross-validation
    for pdb_id, pp_complex in raw_data.items():
        if pp_complex.split is not None:
            data[pp_complex.split][pdb_id] = pp_complex
        else:
            data["train"] = raw_data
    return data


def crossval_split(
    split_data: dict[str, PPDataSet], fold_num: int, num_folds: int
) -> dict[str, list[PPComplex]]:
    """
    number of folds-way cross validation
    @return  dict: split -> [pdb_ids]
    """
    splits = {"train": []}
    # split into train/val/test
    folds = [list(sorted(f)) for f in split_data["train"]]
    val_data = list(sorted(split_data["val"]))
    test_data = list(sorted(split_data["test"]))
    # if val split is pre-specified, do not take fold
    if len(val_data) == 0:
        val_fold = fold_num
        splits["val"] = folds[val_fold]
    else:
        val_fold = -1  # all remaining folds go to train
        splits["val"] = val_data
    # if test split is pre-specified, do not take fold
    if len(test_data) > 0:
        test_fold = val_fold  # must specify to allocate train folds
        splits["test"] = test_data
    # otherwise both val/test labels depend on fold_num
    else:
        test_fold = (fold_num + 1) % num_folds
        splits["test"] = folds[test_fold]
    # add remaining to train
    for idx in range(num_folds):
        if idx in [val_fold, test_fold]:
            continue
        splits["train"].extend(folds[idx])
    return splits


def get_mask(lens):
    """torch.MHA style mask (False = attend, True = mask)"""
    mask = torch.arange(max(lens))[None, :] >= lens[:, None]
    return mask


def find_smallest_diviser(n: int) -> int:
    """
    Returns smallest diviser > 1
    """
    i = 2
    while n % i != 0:
        i += 1
    return i


def find_largest_diviser_smaller_than(n: int, upper: int) -> int:
    """
    Returns largest diviser of n smaller than upper
    """
    i = upper
    while n % i != 0:
        i = i - 1
    return i
