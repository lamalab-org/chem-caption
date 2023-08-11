# -*- coding: utf-8 -*-

"""Global requirements for modular testing."""

import os
from random import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from selfies import encoder

from chemcaption.featurize.text_utils import QA_TEMPLATES, TEXT_TEMPLATES, inspect_info
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

# Implemented utilities for testing

__all__ = [
    "extract_molecule_properties",
    "batch_molecule_properties",
    "extract_representation_strings",
    "convert_molecule",
    "extract_info",
    "fill_template",
    "generate_prompt_test_data",
]

"""Test data."""

BASE_DIR = os.getcwd().split("featurize")[0]
BASE_DIR = BASE_DIR if "tests" in os.getcwd() else os.path.join(os.getcwd(), "tests")

# Sources of truth
PROPERTY_BANK = pd.read_csv(
    os.path.join(BASE_DIR, "data", "pubchem_response.csv")
).drop_duplicates()

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
        property_bank (pd.DataFrame): Dataframe containing molecular properties.
        representation_name (str): Name of molecular representation system.
        property (Union[List[str], str]): Properties of interest. Must be a feature(s) in `property_bank`.

    Returns:
        properties (List[Tuple[str, np.array]]): List of (molecular_string, property value) tuples.
    """
    representation_name = representation_name.lower()
    property = [property] if not isinstance(property, list) else property

    property_bank = property_bank.dropna(axis=0, how="any", subset=[representation_name] + property)

    string_list, property_list = (
        property_bank[representation_name].values.tolist(),
        property_bank[property].values,
    )

    properties = [(k, v) for k, v in zip(string_list, property_list)]

    return properties


def batch_molecule_properties(
    property_bank: pd.DataFrame,
    representation_name: str = "smiles",
    property: Union[List[str], str] = "molar_mass",
    batch_size: int = 2,
) -> List[List[Tuple[str, np.array]]]:
    """Batch extracted SMILES strings and `property` values. Especially useful for `Comparator` testing.

    Args:
        property_bank (pd.DataFrame): Dataframe containing molecular properties.
        representation_name (str): Name of molecular representation system.
        property (Union[List[str], str]): Properties of interest. Must be a feature(s) in `property_bank`.
        batch_size (int): Number of times to batch extracted properties. Defaults to `2`.

    Returns:
        properties (List[List[Tuple[str, np.array]]]): List containing multiple (molecular_string, property value) tuples.
    """
    results = extract_molecule_properties(
        property_bank=property_bank,
        representation_name=representation_name,
        property=property,
    )

    properties = [sorted(results, key=lambda x: random()) for _ in range(batch_size)]

    properties = [k for k in zip(*properties)]

    return properties


def extract_representation_strings(
    molecular_bank: pd.DataFrame,
    in_: str = "smiles",
    out_: str = "selfies",
) -> List[Tuple[str, str]]:
    """Extract molecule representation strings from data bank.

    Args:
        molecular_bank (pd.DataFrame): Dataframe containing molecular information.
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


def convert_molecule(
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
    elif to_kind == "smiles":
        representation_string = Chem.MolToSmiles(molecule.rdkit_mol)
    elif to_kind == "selfies":
        representation_string = encoder(Chem.MolToSmiles(molecule.rdkit_mol))
    else:
        raise ValueError(f"Invalid representation system: {to_kind}.")
    return DISPATCH_MAP[to_kind](representation_string)


def extract_info(
    property_bank: pd.DataFrame,
    representation_name: str,
    property: Union[str, List[str]],
) -> List[dict]:
    """
    Extract molecular information and structure as list of dictionaries.

    Args:
        property_bank (pd.DataFrame): Dataframe containing molecular properties.
        representation_name (str): Name of molecular representation system.
        property (Union[List[str], str]): Properties of interest. Must be a feature(s) in `property_bank`.

    Returns:
        properties (List[dict]): List of (molecular_string, property value) tuples.
    """
    results = extract_molecule_properties(
        property_bank=property_bank, representation_name=representation_name, property=property
    )

    if isinstance(property, (list, tuple)):
        num_features = len(property)
    else:
        num_features = 1
        property = [property]

    results = [
        dict(
            PROPERTY_NAME=property,
            PROPERTY_VALUE=v.reshape(
                num_features,
            ).tolist(),
            REPR_SYSTEM=representation_name,
            REPR_STRING=k,
            PRECISION=4,
            PRECISION_TYPE="decimal",
        )
        for (k, v) in results
    ]
    return results


def fill_template(template: str, bank: List[dict]) -> List[str]:
    """
    Format template and return formatted result.

    Args:
        template (str): Template to format.
        bank (List[dict]): List of dictionaries containing molecular information.

    Returns:
        results (List[str]): List of formatted templates for each dictionary of molecular information.
    """
    results = [template.format(**inspect_info(info)) for info in bank]
    return results


def generate_prompt_test_data(
    property_bank: pd.DataFrame,
    representation_name: str,
    property: Union[str, List[str]],
    key: str = "single",
) -> List[Tuple[dict, str, str]]:
    """
    Generate prompt-related inputs and outputs for testing.

    Args:
        property_bank (pd.DataFrame): Dataframe containing molecular properties.
        representation_name (str): Name of molecular representation system.
        property (Union[List[str], str]): Properties of interest. Must be a feature(s) in `property_bank`.
        key (str): Cardinality of molecular features.

    Returns:
        results (List[Tuple[dict, str, str]]): List of (molecular_string, property value) tuples.
    """
    bank = extract_info(property_bank, representation_name, property)
    templates = (
        QA_TEMPLATES["single"] + TEXT_TEMPLATES["single"]
        if key == "single"
        else QA_TEMPLATES["multiple"] + TEXT_TEMPLATES["multiple"]
    )

    results = [(mol, t, t.format(**inspect_info(mol))) for mol in bank for t in templates]

    return results
