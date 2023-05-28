# -*- coding: utf-8 -*-

"""Tests for chemcaption.molecules subpackage."""

import random

from pandas import read_csv
from rdkit import Chem
from selfies import decoder, encoder

from chemcaption.featurizers import MoleculeFeaturizer
from chemcaption.molecules import Molecule

data_bank = read_csv("data/molecular_bank.csv", encoding="latin-1")
molecular_names, molecular_smiles = (
    data_bank["name"].values.tolist(),
    data_bank["smiles"].values.tolist(),
)



molecular_bank = {
    k: {
        "smiles": v,
        "selfies": encoder(v),
        "smiled_selfies": decoder(encoder(v)),
        "inchi": Chem.MolToInchi(Chem.MolFromSmiles(v)),
    }
    for k, v in zip(molecular_names, molecular_smiles)
}
print(molecular_bank)

if __name__ == "__main__":
    prob = 0.3

    featurizer = MoleculeFeaturizer()

    if prob > 0.5:
        inchi = "InChI=1S/C6H5NO2/c8-7(9)6-4-2-1-3-5-6/h1-5H"
        smiles = "CCC(Cl)C=C"
        selfies_form = encoder(smiles)
        repr_type = "inchi"
        mol = Molecule(inchi, repr_type)

        mol_info = featurizer.get_elements_info(molecule=mol)

        print(f"SMILES: {smiles}")
        print(f"SMILES -> SELFIES -> SMILES: {decoder(encoder(smiles))}")

        bond_type = "SINGLE"
        print(
            f"\n>>> Number of {bond_type.capitalize()} bonds: ",
            featurizer.count_bonds(mol, bond_type=bond_type.upper()),
        )
        print(f"\n>>> Molar mass by featurizer = {featurizer.get_molar_mass(atomic_info=mol_info)}")
        print(f"\n>>> Bond distribution: {featurizer.get_bond_distribution(molecule=mol,)}")
        print(f"\n>>> Bond types: {featurizer.get_unique_bond_types(molecule=mol, )}")
        print("\n>>> Featurizer results: ", featurizer.featurize(mol))
        print(
            "\n>>> Information on all elements in molecule: ",
            featurizer.get_elements_info(
                mol,
            ),
        )

    else:
        molecular_info = {
            "InChI=1S/C6H5NO2/c8-7(9)6-4-2-1-3-5-6/h1-5H": "inchi",
            "InChI=1S/C5H10O4/c6-2-1-4(8)5(9)3-7/h2,4-5,7-9H,1,3H2/t4-,5+/m0/s1": "inchi",
            "CCC(Cl)C=C": "smiles",
            "C(C=O)[C@@H]([C@@H](CO)O)O": "smiles",
            encoder("CCC(Cl)C=C"): "selfies",
        }

        mols = [Molecule(k, v) for k, v in molecular_info.items()]
        index = random.randint(0, len(mols) - 1)
        index = 1

        print("\n>>> Featurizer results: ", featurizer.featurize(molecules=mols)[index])
        print(f"Molecule {index} is represented by: ", mols[index].repr_string)

        element = "n"
        print(
            f"Element {element} appears {featurizer.get_element_frequency(molecule=mols[index],element=element)} times"
        )
        print(mols[index].get_name())

        print(f"Molecule {index} has {featurizer.count_rotable_bonds(mols[index])} rotable bonds.")
