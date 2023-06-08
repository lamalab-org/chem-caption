import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Lipinski
import pandas as pd
import numpy as np

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
        "num_rotable",
        "num_non_rotable",
        "rotable_proportion",
        "non_rotable_proportion",
        "num_hdonors",
        "num_hacceptors"
    ]
    mol = Chem.rdmolops.AddHs(Chem.MolFromSmiles(string))
    num_bonds = len(mol.GetBonds())
    rotable_strict = rdMolDescriptors.CalcNumRotatableBonds(mol, strict=False)
    rotable_non_strict = rdMolDescriptors.CalcNumRotatableBonds(mol, strict=False)

    non_rotable_strict = num_bonds - rotable_strict
    non_rotable_non_strict = num_bonds - rotable_non_strict

    num_donors = Lipinski.NumHDonors(mol)
    num_acceptors = Lipinski.NumHAcceptors(mol)

    values = [
        string,
        num_bonds,
        rotable_strict,
        non_rotable_strict,
        rotable_strict / num_bonds,
        non_rotable_strict / num_bonds,
        num_donors,
        num_acceptors
    ]

    return dict(zip(keys, values))

data = [generate_info(string) for string in smiles_list]

data = pd.DataFrame(data = data)

data.to_csv("data/generated_data.csv", index=False)
print(data.head())

PROPERTY_SUBSET = PROPERTY_BANK.drop(labels = [col for col in PROPERTY_BANK.columns if col.__contains__("num")], axis=1)

print(PROPERTY_SUBSET.head())

NEW_DATA = pd.merge(left = PROPERTY_SUBSET, right = data, left_on="smiles", right_on="smiles")

print(NEW_DATA.head())

NEW_DATA.to_csv("data/merged_pubchem_response.csv", index=False)

print(NEW_DATA.drop_duplicates().shape)
