# -*- coding: utf-8 -*-

"""Global requirements for modular testing."""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from selfies import encoder

from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

"""Test data."""


MOLECULAR_BANK = pd.read_json("data/molecular_bank.json", orient="index").drop_duplicates()
PROPERTY_BANK = pd.read_csv("data/merged_pubchem_response.csv").drop_duplicates()


DISPATCH_MAP = {
    "smiles": SMILESMolecule,
    "selfies": SELFIESMolecule,
    "inchi": InChIMolecule,
}


"""Utility functions."""


def extract_molecule_properties(
    property_bank: pd.DataFrame,
    representation_name: str = "smiles",
    property: Union[List[str], str] = "molar_mass",
) -> List[Tuple[str, np.array]]:
    """Extract SMILES string and the value of `property`.

    Args:
        property_bank (pd.DataFrame): Dataframe containig molecular properties.
        representation_name (str): Name of molecular representation system.
        property (Union[List[str], str]): Properties of interest. Must be a feature(s) in `property_bank`.

    Returns:
        properties (List[Tuple[str, np.array]]): List of (SMILES, property value) tuples.
    """
    representation_name = representation_name.lower()
    property = [property] if not isinstance(property, list) else property

    property_bank = property_bank.dropna(axis=0, how="any", subset=[representation_name] + property)

    string_list, property_list = (
        property_bank[representation_name].values.tolist(),
        property_bank[property].values.tolist(),
    )

    properties = [(k, np.array([v])) for k, v in zip(string_list, property_list)]

    return properties


def extract_representation_strings(
    molecular_bank: pd.DataFrame,
    in_: str = "smiles",
    out_: str = "selfies",
) -> List[Tuple[str, str]]:
    """Extract molecule representation strings from data bank.

    Args:
        molecular_bank (pd.DataFrame): Daraframe containing molecular information.
        in_ (str): Input representation type.
        out_ (str): Output representation type.

    Returns:
        input_output (List[Tuple[str, str]): List of (in_, out_) tuples.
    """
    in_, out_ = in_.lower(), out_.lower()

    molecular_bank = molecular_bank.dropna(axis=0, how="any", subset=[in_, out_])
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
        if to_kind == "selfies":
            representation_string = encoder(representation_string)

    return DISPATCH_MAP[to_kind](representation_string)
