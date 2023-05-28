# -*- coding: utf-8 -*-

"""Utility imports."""
from abc import ABC, abstractmethod

from rdkit import Chem
from selfies import decoder

"""Abstract classes."""


class MoleculeBase(ABC):
    """Base class for molecular representation."""

    def __init__(self, repr_type):
        """Instantiate base class for molecular representation."""
        self.repr_type = repr_type

    @abstractmethod
    def get_rdkit_mol(self):
        """Get molecular representation via rdkit."""
        raise NotImplementedError

    @abstractmethod
    def reveal_hydrogens(self, **kwargs):
        """Reveal hydrogen atoms in molecular structure."""
        raise NotImplementedError

    def get_name(self):
        """Return name of molecule string representation system."""
        return self.repr_type


"""
Lower level Molecule classes

1. SMILESMolecule
2. SELFIESMolecule
3. InChIMolecule
"""


class SMILESMolecule:
    """Lower level molecular representation for SMILES string representation."""

    def __init__(self, smiles):
        """Initialize class."""
        self.smiles = smiles

    def get_rdkit_mol(self):
        """Get rdkit molecular representation from SMILES string."""
        return Chem.MolFromSmiles(self.smiles)


class SELFIESMolecule:
    """Lower level molecular representation for SELFIES string representation."""

    def __init__(self, selfies):
        """Initialize class."""
        self.selfies = selfies
        self.smiles_rep = decoder(selfies)

    def get_rdkit_mol(self):
        """Get rdkit molecular representation from SELFIES string."""
        return Chem.MolFromSmiles(self.smiles_rep)


class InChIMolecule:
    """Lower level molecular representation for InChI string representation."""

    def __init__(self, inchi):
        """Initialize class."""
        self.inchi = inchi

    def get_rdkit_mol(self):
        """Get rdkit molecular representation from InChI string."""
        return Chem.MolFromInchi(self.inchi)


"""Mega classes."""


class Molecule(MoleculeBase):
    """Higher order molecular representation."""

    def __init__(self, repr_string, repr_type="smiles"):
        """Initialize class."""
        super(Molecule, self).__init__(**dict(repr_type=repr_type))
        self.repr_string = repr_string

        self.DISPATCH_MAP = {
            "smiles": SMILESMolecule,
            "selfies": SELFIESMolecule,
            "inchi": InChIMolecule,
        }

        self.molecule = self.DISPATCH_MAP[self.repr_type](self.repr_string)
        self.rdkit_mol = self.get_rdkit_mol()

    def get_rdkit_mol(self):
        """
        Obtain RDKit molecular representation.

        Args:
            None

        Returns:
            None
        """
        return self.molecule.get_rdkit_mol()

    def get_atoms(self):
        """
        Return atomic representation for all atoms present in molecule.

        Args:
            None

        Returns:
            Sequence[Atom]: Sequence of atoms in `molecule`.
        """
        return self.rdkit_mol.GetAtoms()

    def reveal_hydrogens(self, **kwargs):
        """
        Explicitly represent hydrogen atoms in molecular structure.

        Args:
            **kwargs (dict): Keyword arguments.

        Returns:
            None
        """
        self.rdkit_mol = Chem.rdmolops.AddHs(self.rdkit_mol, **kwargs)
        return None
