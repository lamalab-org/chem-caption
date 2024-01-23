# -*- coding: utf-8 -*-

"""Script for generating test data."""

import os
from typing import List

import pandas as pd

from chemcaption.featurize.base import Comparator
from chemcaption.featurize.comparator import IsomorphismComparator
from chemcaption.molecules import SMILESMolecule

BASE_DIR = os.getcwd().replace("legacy", "")

# MOLECULAR_BANK = pd.read_json(os.path.join(BASE_DIR, "molecular_bank.json"), orient="index")
PROPERTY_BANK = pd.read_csv(os.path.join(BASE_DIR, "pubchem_response.csv"))

smiles_list = PROPERTY_BANK["smiles"]


def generate_dataframe(smiles: List[str], comparator: Comparator):
    """Generate a DataFrame containing comparison features between molecules using `comparator`

    Args:
        smiles (List[str]): List of SMILES strings.
        comparator (Comparator): Comparator instance.

    Returns:
        pd.DataFrame: DataFrame containing comparison features. Shape `(N × N)`:
            where `N` = Number of strings in `smiles`.
    """
    df = [
        [
            comparator.compare(
                molecules=[SMILESMolecule(outer_string), SMILESMolecule(inner_string)]
            ).item()
            for inner_string in smiles
        ]
        for outer_string in smiles
    ]
    return pd.DataFrame(data=df, columns=smiles, index=smiles)


def generate_comparator_info(strings: List[str]):
    """
    Return generated profile for a SMILES string.

    Args:
        strings (List[str]): SMILES string.

    Returns:
        Dict[str, Union[int, float]]: Hash map from property name to property value of type int or float.
    """
    keys = [
        "smiles",
        "weisfeiler_lehman_hash",
        "num_bonds",
        "num_rotable_bonds",
        "num_non_rotable_bonds",
        "num_rotable_bonds_strict",
        "num_non_rotable_bonds_strict",
        "rotable_proportion",
        "non_rotable_proportion",
        "rotable_proportion_strict",
        "non_rotable_proportion_strict",
        "num_hydrogen_bond_donors",
        "num_hydrogen_bond_acceptors",
        "num_valence_electrons",
        "num_lipinski_violations",
    ]
    values = ["jd"]

    return dict(zip(keys, values))


if __name__ == "__main__":
    comparator = IsomorphismComparator()
    smiles = PROPERTY_BANK["smiles"].tolist()
    print(generate_dataframe(smiles, comparator))
