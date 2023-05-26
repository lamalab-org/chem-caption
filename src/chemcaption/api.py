# -*- coding: utf-8 -*-

"""Main code."""
import rdkit
from rdkit import Chem
from abc import abstractmethod, ABC
from fastcore.all import typedispatch
from multipledispatch import dispatch
from selfies import decoder, encoder


class MoleculeBase(ABC):
    @abstractmethod
    def get_rdkit_mol(self):
        raise NotImplementedError


class SMILESMolecule(MoleculeBase):
    def __init__(self, smiles):
        self.smiles = smiles

    def get_rdkit_mol(self):
        return Chem.MolFromSmiles(self.smiles)


class SELFIESMolecule(MoleculeBase):
    def __init__(self, selfies):
        self.selfies = selfies
        self.smiles_rep = decoder(selfies)

    def get_rdkit_mol(self):
        return Chem.MolFromSmiles(self.smiles_rep)


class InChIMolecule(MoleculeBase):
    def __init__(self, inchi):
        self.inchi = inchi

    def get_rdkit_mol(self):
        return Chem.MolFromInchi(self.inchi)


class Molecule(MoleculeBase):
    def __init__(self, base_string, rep_type="smiles"):
        self.base_string = base_string
        self.rep_type = rep_type.lower()
        self.dispatch_map = {
            "smiles" : SMILESMolecule,
            "selfies" : SELFIESMolecule,
            "inchi" : InChIMolecule,
        }
        self.molecule = self.dispatch_map[self.rep_type](self.base_string)
    def get_rdkit_mol(self):
        return self.molecule.get_rdkit_mol()

    def get_name(self):
        return self.rep_type
