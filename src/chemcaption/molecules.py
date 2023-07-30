# -*- coding: utf-8 -*-

"""Utility imports."""
from abc import ABC, abstractmethod

import networkx as nx
import rdkit
from rdkit import Chem
from selfies import decoder

# Implemented molecular representation classes.

__all__ = [
    "MoleculeGraph",
    "MoleculeBase",
    "SMILESMolecule",
    "SELFIESMolecule",
    "InChIMolecule",
]


"""Graph representation"""


class MoleculeGraph(nx.Graph):
    def __init__(self, molecule: Chem.Mol):
        super().__init__()

        self.molecule = molecule
        self.periodic_table = rdkit.Chem.GetPeriodicTable()
        self.graph = self.molecule_to_graph()
        self._hash = None

    def molecule_to_graph(self):
        graph = nx.Graph()

        # Generate nodes

        nodes = [
            (
                atom.GetIdx(),
                {
                    "atomic_mass": self.periodic_table.GetAtomicWeight(atom.GetAtomicNum()),
                    "atomic_num": atom.GetAtomicNum(),
                    "atom_symbol": self.periodic_table.GetElementSymbol(atom.GetAtomicNum()),
                },
            )
            for atom in self.molecule.GetAtoms()
        ]

        # Generate edges

        edges = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {"bond_type": bond.GetBondType()})
            for bond in self.molecule.GetBonds()
        ]

        # Store nodes and edges in graph

        graph.add_nodes_from(nodes_for_adding=nodes)
        graph.add_edges_from(ebunch_to_add=edges)

        return graph

    def weisfeiler_lehman_graph_hash(self):
        """Return graph hash according to Weisfeiler-Lehman isomorphism test.

        Args:
            None

        """
        if not self._hash:
            self._hash = nx.weisfeiler_lehman_graph_hash(self.graph)
        return self._hash


"""Abstract classes."""


class MoleculeBase(ABC):
    """Base class for molecular representation."""

    def __init__(
        self,
    ):
        """Instantiate base class for molecular representation."""
        self._rdkit_mol = None
        self.representation_string = None

    @abstractmethod
    def get_rdkit_mol(self):
        """Get molecular representation via rdkit. To be implemented via subclasses."""
        raise NotImplementedError

    @property
    def rdkit_mol(self) -> Chem.Mol:
        """Get molecular representation via rdkit. Getter method."""
        return self._rdkit_mol

    @rdkit_mol.setter
    def rdkit_mol(self, **kwargs: dict) -> None:
        """Set molecular representation via rdkit."""
        self._rdkit_mol = self.get_rdkit_mol()
        return

    def __repr__(self) -> str:
        """Return string representation of molecule object."""
        return f"{self.__class__.__name__}(REPRESENTATION = '{self.representation_string}')"

    def get_atoms(self, hydrogen=True, **kwargs):
        """
        Return atomic representation for all atoms present in molecule.

        Args:
            hydrogen (bool): Reveal hydrogen atoms or not.
            **kwargs (dict): Keyword arguments.

        Returns:
            Sequence[Atom]: Sequence of atoms in `molecule`.
        """
        return self.reveal_hydrogens(**kwargs).GetAtoms() if hydrogen else self.rdkit_mol.GetAtoms()

    def reveal_hydrogens(self, **kwargs: dict) -> Chem.Mol:
        """
        Explicitly represent hydrogen atoms in molecular structure.

        Args:
            **kwargs (dict): Keyword arguments.

        Returns:
            (Chem.Mol): RDKit molecular object with explicit hydrogens.
        """
        return Chem.rdmolops.AddHs(self.rdkit_mol, **kwargs)

    def get_composition(self) -> str:
        """Get composition of molecule."""
        return Chem.rdMolDescriptors.CalcMolFormula(self.rdkit_mol)

    def to_graph(self) -> MoleculeGraph:
        """Convert molecule to graph."""
        graph = MoleculeGraph(molecule=self.reveal_hydrogens())
        return graph


"""Lower level Molecule classes"""


class SMILESMolecule(MoleculeBase):
    """Lower level molecular representation for SMILES string representation."""

    def __init__(self, representation_string: str):
        """Initialize class."""
        super().__init__()
        self.representation_string = Chem.CanonSmiles(representation_string)
        self._rdkit_mol = self.get_rdkit_mol()

    def get_rdkit_mol(self) -> Chem.Mol:
        """Get rdkit molecular representation from SMILES string."""
        return Chem.MolFromSmiles(self.representation_string)


class SELFIESMolecule(MoleculeBase):
    """Lower level molecular representation for SELFIES string representation."""

    def __init__(self, representation_string: str):
        """Initialize class."""
        super().__init__()
        self.representation_string = representation_string
        self.smiles_rep = decoder(representation_string)

        self._rdkit_mol = self.get_rdkit_mol()

    def get_rdkit_mol(self) -> Chem.Mol:
        """Get rdkit molecular representation from SELFIES string."""
        return Chem.MolFromSmiles(self.smiles_rep)


class InChIMolecule(MoleculeBase):
    """Lower level molecular representation for InChI string representation."""

    def __init__(self, representation_string: str):
        """Initialize class."""
        super().__init__()
        self.representation_string = representation_string
        self._rdkit_mol = self.get_rdkit_mol()

    def get_rdkit_mol(self) -> Chem.Mol:
        """Get rdkit molecular representation from InChI string."""
        return Chem.MolFromInchi(self.representation_string)
