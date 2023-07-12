from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Self

import torch
from Bio.PDB import PDBParser, Structure
from torch import Tensor
from torch_cluster import knn_graph
from torch_geometric.data import HeteroData


class AminoAcid(StrEnum):
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


class Element(StrEnum):
    HYDROGEN = "H"
    CARBON = "C"
    NITROGEN = "N"
    OXYGEN = "O"
    SULPHUR = "S"


@dataclass
class Atom:
    name: str
    element: Element
    pos: Tensor
    charge: Optional[float] = None
    beta: Optional[float] = None

    @classmethod
    def construct_atom(cls, atom) -> Self:
        pos = torch.from_numpy(atom.get_coord())
        name = atom.get_name()
        element = atom.element
        return cls(name=name, element=Element(element), pos=pos)


@dataclass
class Residue:
    name: AminoAcid
    idx: int
    atoms: list[Atom]

    @classmethod
    def construct_residue(cls, residue, idx: int, atoms: list[Atom]) -> Self:
        name = AminoAcid(residue.get_resname())
        return cls(name, idx, atoms)

    @property
    def elements(self) -> list[Element]:
        return [atom.element for atom in self.atoms]


@dataclass
class Protein:
    sequence: list[Residue] = field(default_factory=list)
    filtered_sequence: list[Residue] = field(default_factory=list)
    tokenized_sequence: list[int] = field(default_factory=list)
    n_i_feat: Optional[Tensor] = None
    u_i_feat: Optional[Tensor] = None
    v_i_feat: Optional[Tensor] = None

    def __len__(self):
        return len(self.sequence)

    def append(self, residue: Residue) -> None:
        self.sequence.append(residue)

    @property
    def sequence_names(self) -> list[AminoAcid]:
        return [residue.name for residue in self.sequence]

    @property
    def sequence_str(self) -> str:
        return "".join([amino_acid_symbols[residue.name] for residue in self.sequence])

    @property
    def filtered_sequence_names(self) -> list[AminoAcid]:
        return [residue.name for residue in self.filtered_sequence]

    @property
    def filtered_sequence_str(self) -> str:
        return "".join(
            [amino_acid_symbols[residue.name] for residue in self.filtered_sequence]
        )

    @property
    def all_elements(self) -> list[Element]:
        return [element for residue in self.sequence for element in residue.elements]

    @property
    def filtered_elements(self) -> list[Element]:
        return [
            element
            for residue in self.filtered_sequence
            for element in residue.elements
        ]

    @classmethod
    def make_protein(cls, structure: Structure.Structure) -> Self:
        protein = cls()
        residues = structure.get_residues()
        for i, residue in enumerate(residues):
            atoms = [Atom.construct_atom(atom) for atom in residue]
            res = Residue.construct_residue(residue, i, atoms)
            protein.append(res)

        return protein

    @property
    def all_positions(self) -> Tensor:
        positions = [atom.pos for residue in self.sequence for atom in residue.atoms]
        return torch.stack(positions)

    @property
    def filtered_positions(self) -> Tensor:
        positions = [
            atom.pos for residue in self.filtered_sequence for atom in residue.atoms
        ]
        return torch.stack(positions)

    def filter(self, atoms_to_keep: list[str]) -> None:
        for residue in self.sequence:
            self.filtered_sequence.append(
                Residue(
                    residue.name,
                    residue.idx,
                    [atom for atom in residue.atoms if atom.name not in atoms_to_keep],
                )
            )

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


@dataclass
class PPComplex:
    id: str
    receptor: Protein
    ligand: Protein
    split: Optional[str] = None
    graph: HeteroData = HeteroData()

    def populate_graph(self, use_orientation_features: bool, knn_size: int) -> None:
        self.graph["name"] = self.id
        # retrieve position and compute kNN
        for protein, name in zip([self.receptor, self.ligand], ["receptor", "ligand"]):
            self.graph[name].pos = protein.filtered_positions.float()
            self.graph[name].x = protein.filtered_sequence  # _seq is residue id
            if use_orientation_features:
                self.graph[name].n_i_feat = protein.n_i_feat.float()
                self.graph[name].u_i_feat = protein.u_i_feat.float()
                self.graph[name].v_i_feat = protein.v_i_feat.float()
            # kNN graph
            edge_index = knn_graph(self.graph[name].pos, knn_size)
            self.graph[name, "contact", name].edge_index = edge_index
        # center receptor at origin
        center = self.graph["receptor"].pos.mean(dim=0, keepdim=True)
        for key in ["receptor", "ligand"]:
            self.graph[key].pos = self.graph[key].pos - center
        self.graph.center = center  # save old center


def parse_pdb(file: str, name: str) -> Protein:
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(name, file)

    return Protein.make_protein(structure)


if __name__ == "__main__":
    file = "/Users/danieljackson/target.pdb"
    test_protein = parse_pdb(file, "test")

    # print(getattr(test_protein, ("sequence_names")))

    # print([residue.elements for residue in test_protein.sequence])

    tokenizer = {"C": 1, "N": 2, "O": 3, "S": 4, "H": 5}

    # test_protein.tokenize_sequence(tokenizer)
    # print(test_protein.tokenized_sequence)

    print(
        [element for residue in test_protein.sequence for element in residue.elements]
    )
