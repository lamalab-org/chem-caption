# -*- coding: utf-8 -*-

"""Utility imports."""
from rdkit import Chem

from abc import abstractmethod, ABC
from selfies import decoder

"""Abstract classes."""


class MoleculeBase(ABC):
    def __init__(self, repr_type):
        self.repr_type = repr_type

    @abstractmethod
    def get_rdkit_mol(self):
        raise NotImplementedError

    @abstractmethod
    def reveal_hydrogens(self, **kwargs):
        raise NotImplementedError

    def get_name(self):
        return self.repr_type


"""
Lower level Molecule classes

1. SMILESMolecule
2. SELFIESMolecule
3. InChIMolecule
"""


class SMILESMolecule:
    """
    Lower level molecular representation for SMILES string representation.
    """
    def __init__(self, smiles):
        self.smiles = smiles

    def get_rdkit_mol(self):
        return Chem.MolFromSmiles(self.smiles)


class SELFIESMolecule:
    """
    Lower level molecular representation for SELFIES string representation.
    """
    def __init__(self, selfies):
        self.selfies = selfies
        self.smiles_rep = decoder(selfies)

    def get_rdkit_mol(self):
        return Chem.MolFromSmiles(self.smiles_rep)


class InChIMolecule:
    """
    Lower level molecular representation for InChI string representation.
    """
    def __init__(self, inchi):
        self.inchi = inchi

    def get_rdkit_mol(self):
        return Chem.MolFromInchi(self.inchi)


"""Mega classes."""


class Molecule(MoleculeBase):
    """
    Higher order molecular representation.
    """
    def __init__(self, repr_string, repr_type="smiles"):
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
