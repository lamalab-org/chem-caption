# -*- coding: utf-8 -*-

"""Global requirements for modular testing."""

from typing import Union

import numpy as np
import pandas as pd
from rdkit import Chem
from selfies import encoder

from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

"""Test data."""


MOLECULAR_BANK = pd.read_json("data/molecular_bank.json", orient="index")
PROPERTY_BANK = pd.read_csv("data/pubchem_response.csv")


DISPATCH_MAP = {
    "smiles": SMILESMolecule,
    "selfies": SELFIESMolecule,
    "inchi": InChIMolecule,
}

NAMES = [
    "MolecularWeight",
    "ExactMass",
    "MonoisotopicMass",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "smiles",
]

NEW_NAMES = [
    "molar_mass",
    "exact_mass",
    "monoisotopic_mass",
    "num_hdonors",
    "num_hacceptors",
    "num_rotable",
    "smiles",
]

NAME_DICT = dict(zip(NAMES, NEW_NAMES))
PROPERTY_BANK.rename(columns=NAME_DICT, inplace=True)
PROPERTY_BANK = PROPERTY_BANK.loc[:, NEW_NAMES]
"""Utility functions."""

def extract_molecule_properties(property_bank, property="molar_mass"):
    """Extract SMILES string and the value of `property`."""
    smiles_list, property_list = (
        property_bank["smiles"].values.tolist(),
        property_bank[property].values.tolist(),
    )
    properties = [(k, np.array([v])) for k, v in zip(smiles_list, property_list) if not np.isnan(v)]

    return properties


def extract_representation_strings(
    molecular_bank: pd.DataFrame, in_: str = "smiles", out_: str = "selfies"
):
    """Extract molecule representation strings from data bank."""
    in_, out_ = in_.lower(), out_.lower()
    in_list, out_list = molecular_bank[in_].tolist(), molecular_bank[out_].tolist()
    input_output = [(k, v) for k, v in zip(in_list, out_list)]
    return input_output


def _convert_molecule(
    molecule: Union[InChIMolecule, SELFIESMolecule, SMILESMolecule], to_kind: str = "smiles"
) -> Union[InChIMolecule, SELFIESMolecule, SMILESMolecule]:
    """Convert molecules between representational systems.

    Args:
        molecule (Union[InChIMolecule, SELFIESMolecule, SMILESMolecule]): Molecular representation instance.
        to_kind (str): Target molecular representation system.

    Returns:
        Union[InChIMolecule, SELFIESMolecule, SMILESMolecule]: New molecular representation instance in `to_kind`
            representation system.
    """
    to_kind = to_kind.lower()

    if to_kind == "inchi":
        representation_string = Chem.MolToInchi(molecule.rdkit_mol)
    else:
        representation_string = Chem.MolToSmiles(molecule.rdkit_mol)
        representation_string = Chem.CanonSmiles(representation_string)
        if to_kind == "selfies":
            representation_string = encoder(representation_string)

    return DISPATCH_MAP[to_kind](representation_string)
