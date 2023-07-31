from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import copy

import torch
from Bio.PDB import PDBParser, Structure
from Bio.PDB import Atom as BioAtom
from torch import Tensor
from torch_cluster import knn_graph
from torch_geometric.data import HeteroData

# from data.pdb import PDB_LINE
PDB_LINE = "ATOM  {:>5}  {:<3} {} {} {:>3}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00           {}\n"


class AminoAcid(Enum):
    ALA = "ALA"
    ARG = "ARG"
    ASN = "ASN"
    ASP = "ASP"
    CYS = "CYS"
    GLU = "GLU"
    GLN = "GLN"
    GLY = "GLY"
    HIS = "HIS"
    ILE = "ILE"
    LEU = "LEU"
    LYS = "LYS"
    MET = "MET"
    PHE = "PHE"
    PRO = "PRO"
    SER = "SER"
    THR = "THR"
    TRP = "TRP"
    TYR = "TYR"
    VAL = "VAL"


amino_acid_symbols = {
    AminoAcid.ALA: "A",
    AminoAcid.ARG: "R",
    AminoAcid.ASN: "N",
    AminoAcid.ASP: "D",
    AminoAcid.CYS: "C",
    AminoAcid.GLU: "E",
    AminoAcid.GLN: "Q",
    AminoAcid.GLY: "G",
    AminoAcid.HIS: "H",
    AminoAcid.ILE: "I",
    AminoAcid.LEU: "L",
    AminoAcid.LYS: "K",
    AminoAcid.MET: "M",
    AminoAcid.PHE: "F",
    AminoAcid.PRO: "P",
    AminoAcid.SER: "S",
    AminoAcid.THR: "T",
    AminoAcid.TRP: "W",
    AminoAcid.TYR: "Y",
    AminoAcid.VAL: "V",
}


class Element(Enum):
    HYDROGEN = "H"
    CARBON = "C"
    NITROGEN = "N"
    OXYGEN = "O"
    SULPHUR = "S"
    PHOSPHOROUS = "P"


@dataclass
class Atom:
    name: str
    serial_number: int
    amino_acid: AminoAcid
    chain: str
    res_idx: int
    pos: Tensor
    element: Element
    charge: Optional[float] = None
    beta: Optional[float] = None

    @classmethod
    def construct_atom(
        cls, atom: BioAtom, chain: str, res_idx: int, residue: str
    ) -> Atom:
        pos = torch.from_numpy(atom.get_coord())
        name = atom.get_name()
        amino_acid = AminoAcid(residue)
        serial_number = atom.serial_number
        element = atom.element
        return cls(
            name,
            serial_number,
            amino_acid,
            chain,
            res_idx,
            pos,
            element=Element(element),
        )


@dataclass
class Protein:
    sequence: list[Atom] = field(default_factory=list)
    filtered_sequence: list[Atom] = field(default_factory=list)
    tokenized_sequence: list[int] = field(default_factory=list)
    n_i_feat: Optional[Tensor] = None
    u_i_feat: Optional[Tensor] = None
    v_i_feat: Optional[Tensor] = None

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx: int):
        return self.sequence[idx]

    def append(self, atom: Atom) -> None:
        self.sequence.append(atom)

    @property
    def sequence_names(self) -> list[AminoAcid]:
        return [residue.name for residue in self.sequence]

    @property
    def sequence_str(self) -> str:
        return "".join([amino_acid_symbols[atom.amino_acid] for atom in self.sequence])

    @property
    def filtered_sequence_names(self) -> list[AminoAcid]:
        return [atom.amino_acid for atom in self.filtered_sequence]

    @property
    def filtered_sequence_str(self) -> str:
        return "".join(
            [amino_acid_symbols[atom.amino_acid] for atom in self.filtered_sequence]
        )

    @property
    def all_elements(self) -> list[Element]:
        return [atom.element for atom in self.sequence]

    @property
    def filtered_elements(self) -> list[Element]:
        return [atom.element for atom in self.filtered_sequence]

    @classmethod
    def make_protein(cls, structure: Structure.Structure, chains: str) -> Protein:
        protein = cls()
        for chain in structure.get_chains():
            chain_id = chain.id
            if chain_id in chains:
                for residue in chain.get_residues():
                    if residue.id[0] == " ":
                        for atom in residue:
                            protein.append(
                                Atom.construct_atom(
                                    atom, chain_id, residue.id[1], residue.get_resname()
                                )
                            )

        return protein

    @property
    def all_positions(self) -> Tensor:
        positions = [atom.pos for atom in self.sequence]
        return torch.stack(positions)

    @property
    def filtered_positions(self) -> Tensor:
        positions = [atom.pos for atom in self.filtered_sequence]
        return torch.stack(positions)

    def filter(self, atoms_to_keep: Optional[list[str]]) -> None:
        if atoms_to_keep is not None:
            for atom in self.sequence:
                if atom.name in atoms_to_keep:
                    self.filtered_sequence.append(atom)
        else:
            self.filtered_sequence = self.sequence

    def compute_orientation_vectors(
        self,
        c_alpha_coords_list: list[Tensor],
        n_coords_list: list[Tensor],
        c_coords_list: list[Tensor],
    ) -> None:
        n_i_list: list[Tensor] = []
        u_i_list: list[Tensor] = []
        v_i_list: list[Tensor] = []
        for c_alpha_coord, n_coord, c_coord in zip(
            c_alpha_coords_list, n_coords_list, c_coords_list
        ):
            u_i = (n_coord - c_alpha_coord) / torch.linalg.vector_norm(
                n_coord - c_alpha_coord
            )
            t_i = (c_coord - c_alpha_coord) / torch.linalg.vector_norm(
                c_coord - c_alpha_coord
            )
            n_i = torch.linalg.cross(u_i, t_i) / torch.linalg.vector_norm(
                torch.linalg.cross(u_i, t_i)
            )
            v_i = torch.linalg.cross(n_i, u_i)
            assert (
                torch.abs(torch.linalg.vector_norm(v_i) - 1.0) < 1e-5
            ), "v_i norm is not 1"

            n_i_list.append(n_i)
            u_i_list.append(u_i)
            v_i_list.append(v_i)

        self.n_i_feat = torch.stack(n_i_list)
        self.u_i_feat = torch.stack(u_i_list)
        self.v_i_feat = torch.stack(v_i_list)

        assert self.n_i_feat.shape == self.u_i_feat.shape == self.v_i_feat.shape

    def tokenize_sequence(self, tokenizer: dict[Element, int]) -> None:
        self.tokenized_sequence = [
            tokenizer[element] for element in self.filtered_elements
        ]

    def update(self, new_positions: Tensor):
        i = 0
        for atom in self.filtered_sequence:
            atom.pos = new_positions[i]
            i += 1

    def write_to_pdb(self, file_path: str) -> None:
        lines: list[str] = []
        for atom in self.filtered_sequence:
            lines.append(
                PDB_LINE.format(
                    atom.serial_number,
                    atom.name,
                    atom.amino_acid.value,
                    atom.chain,
                    atom.res_idx,
                    atom.pos[0],
                    atom.pos[1],
                    atom.pos[2],
                    atom.element.value,
                )
            )

        with open(file_path, "a") as file:
            file.writelines(lines)


@dataclass
class PPComplex:
    id: str
    receptor: Protein
    ligand: Protein
    split: Optional[str] = None
    graph: Optional[HeteroData] = None

    def populate_graph(self, use_orientation_features: bool, knn_size: int) -> None:
        graph = HeteroData()
        graph["name"] = self.id
        # retrieve position and compute kNN
        for protein, name in zip([self.receptor, self.ligand], ["receptor", "ligand"]):
            graph[name].pos = protein.filtered_positions.float()
            graph[name].x = [
                atom.amino_acid for atom in protein.filtered_sequence
            ]  # _seq is residue id
            if use_orientation_features:
                graph[name].n_i_feat = protein.n_i_feat.float()
                graph[name].u_i_feat = protein.u_i_feat.float()
                graph[name].v_i_feat = protein.v_i_feat.float()
            # kNN graph
            edge_index = knn_graph(graph[name].pos, knn_size)
            graph[name, "contact", name].edge_index = edge_index
        # center receptor at origin
        center = graph["receptor"].pos.mean(dim=0, keepdim=True)
        for key in ["receptor", "ligand"]:
            graph[key].pos = graph[key].pos - center
        graph.center = center  # save old center
        self.graph = graph

    def update_proteins_from_graph(self, graph: HeteroData) -> None:
        self.graph = graph
        self.receptor.update(graph["receptor"].pos)
        self.ligand.update(graph["ligand"].pos)

    def write_to_pdb(
        self, rec_file_path: str, lig_file_path: Optional[str] = None
    ) -> None:
        self.receptor.write_to_pdb(rec_file_path)
        if lig_file_path is None:
            lig_file_path = rec_file_path
        self.ligand.write_to_pdb(lig_file_path)

    def __str__(self) -> str:
        return f"{self.id}, {id(self)}"


def parse_combined_pdb(
    file: str, name: str, receptor_chains: list[str], ligand_chains: list[str]
) -> tuple[Protein, Protein]:
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(name, file)
    receptor = Protein.make_protein(structure, receptor_chains)
    ligand = Protein.make_protein(structure, ligand_chains)
    return receptor, ligand


def parse_individual_pdb(file: str, name: str, chains: str) -> Protein:
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(name, file)
    protein = Protein.make_protein(structure, chains)
    return protein


if __name__ == "__main__":
    pdb_parser = PDBParser()
    # lig_file = "datasets/single_pair_dataset/structures/1A2K_l_b.pdb"
    rec_file = "datasets/single_pair_dataset/structures/1A2K_r_b.pdb"
    # lig_structure = pdb_parser.get_structure("test", lig_file)
    rec_structure = pdb_parser.get_structure("test", rec_file)
    # test = structure.get_atoms()
    # for atom in test:
    #     print(atom.serial_number)
    #     break
    # ligand = Protein.make_protein(lig_structure)
    receptor = Protein.make_protein(rec_structure)
    receptor.filter(None)
    receptor.write_to_pdb("test_pdb.pdb")
    print(receptor[0])
    # ligand.filter(["CA"])
    # receptor.filter(["CA"])
    # binding_complex = PPComplex("1", receptor, ligand)
    # binding_complex.populate_graph(False, 20)
    # print(len(binding_complex.graph["receptor"].x))
    # print(binding_complex.graph["receptor"].pos.shape)
