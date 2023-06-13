# -*- coding: utf-8 -*-

"""Script for generating test data."""

import pandas as pd
from rdkit.Chem import Lipinski, rdMolDescriptors

from chemcaption.featurizers import (
    ElementCountFeaturizer,
    ElementCountProportionFeaturizer,
    ElementMassFeaturizer,
    ElementMassProportionFeaturizer, SMARTSFeaturizer,
)
from chemcaption.molecules import SMILESMolecule

MOLECULAR_BANK = pd.read_json("data/molecular_bank.json", orient="index")
PROPERTY_BANK = pd.read_csv("data/pubchem_response.csv")

smiles_list = PROPERTY_BANK["smiles"]


def generate_info(string: str):
    """
    Return generated profile for a SMILES string.

    Args:
        string (str): SMILES string.

    Returns:
        (Dict[str, Union[int, float]]): Hash map from property name to property value of type int or float.
    """
    keys = [
        "smiles",
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
    ]
    preset = [
        "Carbon",
        "Hydrogen",
        "Nitrogen",
        "Oxygen",
        "Sulfur",
        "Phosphorus",
        "Fluorine",
        "Chlorine",
        "Bromine",
        "Iodine",
    ]

    mass_featurizer = ElementMassFeaturizer(preset=preset)
    mass_ratio_featurizer = ElementMassProportionFeaturizer(preset=preset)

    count_featurizer = ElementCountFeaturizer(preset=preset)
    count_ratio_featurizer = ElementCountProportionFeaturizer(preset=preset)

    mol = SMILESMolecule(string)

    num_bonds = len(mol.rdkit_mol.GetBonds())
    rotable_strict = rdMolDescriptors.CalcNumRotatableBonds(mol.rdkit_mol, strict=True)
    rotable_non_strict = rdMolDescriptors.CalcNumRotatableBonds(mol.rdkit_mol, strict=False)

    non_rotable_strict = num_bonds - rotable_strict
    non_rotable_non_strict = num_bonds - rotable_non_strict

    num_donors = Lipinski.NumHDonors(mol.rdkit_mol)
    num_acceptors = Lipinski.NumHAcceptors(mol.rdkit_mol)

    masses = mass_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += mass_featurizer.feature_labels()

    mass_ratios = mass_ratio_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += mass_ratio_featurizer.feature_labels()

    counts = count_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += count_featurizer.feature_labels()

    count_ratios = count_ratio_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += count_ratio_featurizer.feature_labels()


    values = [
        string,
        num_bonds,
        rotable_non_strict,
        non_rotable_non_strict,
        rotable_strict,
        non_rotable_strict,
        rotable_non_strict / num_bonds,
        non_rotable_non_strict / num_bonds,
        rotable_strict / num_bonds,
        non_rotable_strict / num_bonds,
        num_donors,
        num_acceptors,
    ]
    values += masses
    values += mass_ratios

    values += counts
    values += count_ratios

    for preset in ['rings', 'organic', "heterocyclic", "warheads", "scaffolds", "amino"]:
        for val in [True, False]:
            smarts_featurizer = SMARTSFeaturizer(count=val, names=preset)
            smarts_presence = smarts_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()

            keys += smarts_featurizer.feature_labels()
            values += smarts_presence


    return dict(zip(keys, values))


data = [generate_info(string) for string in smiles_list]

data = pd.DataFrame(data=data)
# data.to_csv("data/generated_data.csv", index=False)

PROPERTY_SUBSET = PROPERTY_BANK.drop(
    labels=[col for col in PROPERTY_BANK.columns if col.__contains__("num")], axis=1
)

NEW_DATA = pd.merge(left=PROPERTY_SUBSET, right=data, left_on="smiles", right_on="smiles").rename(
    columns={"molar_mass": "molecular_mass"}
)


NEW_DATA.to_csv("data/merged_pubchem_response.csv", index=False)

print(NEW_DATA.columns)
