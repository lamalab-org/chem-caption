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

    def get_name(self):
        return self.repr_type


class SMILESMolecule():
    def __init__(self, smiles):
        self.smiles = smiles

    def get_rdkit_mol(self):
        return Chem.MolFromSmiles(self.smiles)


class SELFIESMolecule():
    def __init__(self, selfies):
        self.selfies = selfies
        self.smiles_rep = decoder(selfies)

    def get_rdkit_mol(self):
        return Chem.MolFromSmiles(self.smiles_rep)


class InChIMolecule():
    def __init__(self, inchi):
        self.inchi = inchi

    def get_rdkit_mol(self):
        return Chem.MolFromInchi(self.inchi)


"""Mega classes."""


class Molecule(MoleculeBase):
    def __init__(self, repr_string, repr_type="smiles"):
        super(Molecule, self).__init__(**dict(repr_type=repr_type))
        self.repr_string = repr_string

        self.dispatch_map = {
            "smiles": SMILESMolecule,
            "selfies": SELFIESMolecule,
            "inchi": InChIMolecule,
        }

        self.molecule = self.dispatch_map[self.repr_type](self.repr_string)
        self.rdkit_mol = self.get_rdkit_mol()

    def get_rdkit_mol(self):
        return self.molecule.get_rdkit_mol()

    def get_atoms(self):
        return self.rdkit_mol.GetAtoms()

    def reveal_hydrogens(self, **kwargs):
        self.rdkit_mol = Chem.rdmolops.AddHs(self.rdkit_mol, **kwargs)
        return None
