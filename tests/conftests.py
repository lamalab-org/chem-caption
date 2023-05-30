# -*- coding: utf-8 -*-

"""Global requirements for modular testing."""

import pandas as pd
from rdkit import Chem
from selfies import encoder

from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

"""Test data."""

data_bank = pd.read_csv("data/molecular_bank.csv", encoding="latin-1")
data_bank["smiles"] = data_bank["smiles"].apply(Chem.CanonSmiles)
molecular_names, molecular_smiles = (
    data_bank["name"].values.tolist(),
    data_bank["smiles"].values.tolist(),
)

"""Constant variables."""

MOLECULAR_BANK = {
    k: {
        "smiles": v,
        "selfies": encoder(v),
        "inchi": Chem.MolToInchi(Chem.MolFromSmiles(v)),
    }
    for k, v in zip(molecular_names, molecular_smiles)
}

DISPATCH_MAP = {
    "smiles": SMILESMolecule,
    "selfies": SELFIESMolecule,
    "inchi": InChIMolecule,
}

"""Utility functions."""


def extract_representation_strings(molecular_bank, in_="smiles", out_="selfies"):
    """Extract molecule representation strings from data bank."""
    in_, out_ = in_.lower(), out_.lower()
    input_output = [(v[in_], v[out_]) for k, v in molecular_bank.items()]
    return input_output


def _convert_molecule(molecule, to_kind="smiles"):
    """Convert molecules between representational systems."""
    to_kind = to_kind.lower()

    if to_kind == "inchi":
        representation_string = Chem.MolToInchi(molecule.rdkit_mol)
    else:
        representation_string = Chem.MolToSmiles(molecule.rdkit_mol)
        representation_string = Chem.CanonSmiles(representation_string)
        if to_kind == "selfies":
            representation_string = encoder(representation_string)

    return DISPATCH_MAP[to_kind](representation_string)
