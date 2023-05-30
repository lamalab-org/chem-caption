# -*- coding: utf-8 -*-

"""Utility imports."""
from abc import ABC, abstractmethod

from rdkit import Chem
from selfies import decoder

"""Abstract classes."""


class MoleculeBase(ABC):
    """Base class for molecular representation."""

    def __init__(
        self,
    ):
        """Instantiate base class for molecular representation."""
        # self.representation_name = representation_name
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
            None

        Returns:
            Sequence[Atom]: Sequence of atoms in `molecule`.
        """
        return self.reveal_hydrogens(**kwargs).GetAtoms() if hydrogen else self.rdkit_mol.GetAtoms()

    def reveal_hydrogens(self, **kwargs) -> Chem.Mol:
        """
        Explicitly represent hydrogen atoms in molecular structure.

        Args:
            **kwargs (dict): Keyword arguments.

        Returns:
            None
        """
        return Chem.rdmolops.AddHs(self.rdkit_mol, **kwargs)


"""
Lower level Molecule classes

1. SMILESMolecule
2. SELFIESMolecule
3. InChIMolecule
"""


class SMILESMolecule(MoleculeBase):
    """Lower level molecular representation for SMILES string representation."""

    def __init__(self, representation_string):
        """Initialize class."""
        super().__init__()
        self.representation_string = Chem.CanonSmiles(representation_string)
        self._rdkit_mol = self.get_rdkit_mol()

    def get_rdkit_mol(self) -> Chem.Mol:
        """Get rdkit molecular representation from SMILES string."""
        return Chem.MolFromSmiles(self.representation_string)


class SELFIESMolecule(MoleculeBase):
    """Lower level molecular representation for SELFIES string representation."""

    def __init__(self, representation_string):
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

    def __init__(self, representation_string):
        """Initialize class."""
        super().__init__()
        self.representation_string = representation_string
        self._rdkit_mol = self.get_rdkit_mol()

    def get_rdkit_mol(self) -> Chem.Mol:
        """Get rdkit molecular representation from InChI string."""
        return Chem.MolFromInchi(self.representation_string)
