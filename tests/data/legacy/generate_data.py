# -*- coding: utf-8 -*-

"""Script for generating test data."""

import gc
import os
from argparse import ArgumentParser

import pandas as pd
from rdkit.Chem import Lipinski, rdMolDescriptors

from chemcaption.featurize.composition import (
    AtomCountFeaturizer,
    ElementCountFeaturizer,
    ElementCountProportionFeaturizer,
    ElementMassFeaturizer,
    ElementMassProportionFeaturizer,
)
from chemcaption.featurize.electronicity import ValenceElectronCountFeaturizer
from chemcaption.featurize.rules import LipinskiViolationCountFeaturizer
from chemcaption.featurize.spatial import (
    AsphericityFeaturizer,
    EccentricityFeaturizer,
    InertialShapeFactorFeaturizer,
    NPRFeaturizer,
    PMIFeaturizer,
)
from chemcaption.featurize.substructure import IsomorphismFeaturizer, SMARTSFeaturizer
from chemcaption.molecules import SMILESMolecule

BASE_DIR = os.getcwd()

MOLECULAR_BANK = pd.read_json(os.path.join(BASE_DIR, "molecular_bank.json"), orient="index")
PROPERTY_BANK = pd.read_csv(os.path.join(BASE_DIR, "pubchem_response.csv"))


def main(args):
    args = args.parse_args()
    persist_data(chunk_size=args.chunk_size)
    return


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
        "weisfeiler_lehman_hash",
        "num_atoms",
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
    lipinski_featurizer = LipinskiViolationCountFeaturizer()

    valence_featurizer = ValenceElectronCountFeaturizer()
    isomorphism_featurizer = IsomorphismFeaturizer()

    atom_count_featurizer = AtomCountFeaturizer()
    npr_featurizer = NPRFeaturizer()
    pmi_featurizer = PMIFeaturizer()
    asphericity_featurizer = AsphericityFeaturizer()
    eccentricity_featurizer = EccentricityFeaturizer()
    inertial_featurizer = InertialShapeFactorFeaturizer()

    mol = SMILESMolecule(string)

    wl_hash = isomorphism_featurizer.featurize(mol).item()

    atom_count = atom_count_featurizer.featurize(mol).item()

    num_bonds = len(mol.rdkit_mol.GetBonds())

    rotable_strict = rdMolDescriptors.CalcNumRotatableBonds(mol.rdkit_mol, strict=True)
    rotable_non_strict = rdMolDescriptors.CalcNumRotatableBonds(mol.rdkit_mol, strict=False)

    non_rotable_strict = num_bonds - rotable_strict
    non_rotable_non_strict = num_bonds - rotable_non_strict

    num_donors = Lipinski.NumHDonors(mol.rdkit_mol)
    num_acceptors = Lipinski.NumHAcceptors(mol.rdkit_mol)

    num_lipinski_violations = lipinski_featurizer.featurize(molecule=mol).item()

    masses = mass_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += mass_featurizer.feature_labels()

    mass_ratios = mass_ratio_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += mass_ratio_featurizer.feature_labels()

    counts = count_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += count_featurizer.feature_labels()

    count_ratios = count_ratio_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += count_ratio_featurizer.feature_labels()

    valence_count = valence_featurizer.featurize(molecule=mol).item()

    values = [
        string,
        wl_hash,
        atom_count,
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
        valence_count,
        num_lipinski_violations,
    ]
    values += masses
    values += mass_ratios

    values += counts
    values += count_ratios

    # print("Done 1!")
    #
    # npr_values = npr_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    # keys += npr_featurizer.feature_labels()
    # values += npr_values
    # print("Done 2!")
    #
    # pmi_values = pmi_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    # keys += pmi_featurizer.feature_labels()
    # values += pmi_values
    # print("Done 3!")
    #
    # asphericity = asphericity_featurizer.featurize(molecule=mol).item()
    # print("Done 4!")
    # eccentricity = eccentricity_featurizer.featurize(molecule=mol).item()
    # print("Done 5!")
    # inertia = inertial_featurizer.featurize(molecule=mol).item()
    # print("Done 6!")
    #
    # keys += asphericity_featurizer.feature_labels()
    # keys += eccentricity_featurizer.feature_labels()
    # keys += inertial_featurizer.feature_labels()
    #
    # values += [asphericity]
    # values += [eccentricity]
    # values += [inertia]

    print("Done 2!")

    for preset in ["rings", "organic", "heterocyclic", "warheads", "scaffolds", "amino"]:
        for val in [True, False]:
            smarts_featurizer = SMARTSFeaturizer(count=val, names=preset)
            smarts_presence = smarts_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()

            keys += smarts_featurizer.feature_labels()
            values += smarts_presence

    gc.collect()

    return dict(zip(keys, values))


def persist_data(chunk_size=30):
    """Break data into chunks and persist."""
    smiles_list = PROPERTY_BANK["smiles"]
    PROPERTY_SUBSET = PROPERTY_BANK.drop(
        labels=[col for col in PROPERTY_BANK.columns if col.__contains__("num")], axis=1
    )

    PROPERTY_SUBSET = PROPERTY_SUBSET.rename(
        columns={
            "molar_mass": "molecular_mass",
            "exact_mass": "exact_molecular_mass",
            "monoisotopic_mass": "monoisotopic_molecular_mass",
        }
    )

    NEW_PATH = os.path.join(BASE_DIR.replace("legacy", ""), "merged_pubchem_response.csv")

    if os.path.isfile(NEW_PATH):
        old_data = pd.read_csv(NEW_PATH)
        start_index = len(old_data)
    else:
        start_index = 0

    print(
        f"Starting from index {start_index} out of {len(smiles_list)}: {start_index}/{len(smiles_list)}..."
    )

    chunks = [
        smiles_list[i : i + chunk_size] for i in range(start_index, len(smiles_list), chunk_size)
    ]

    running_size = start_index

    for chunk in chunks:
        data = [generate_info(string) for string in chunk]
        data = pd.DataFrame(data=data)
        # data.to_csv("data/merged_pubchem_response_.csv", index=False)

        if os.path.isfile(NEW_PATH):
            old_data = pd.read_csv(NEW_PATH)
            # columns = old_data.columns
            data = pd.concat((old_data, data), axis=0)

        data.to_csv(NEW_PATH, index=False)
        running_size += len(chunk)

        print(f"Persisted {running_size}/{len(smiles_list)} SMILES strings!\n")

        gc.collect()

    print("All SMILES strings processed!\n")

    data = pd.read_csv(NEW_PATH)
    NEW_DATA = pd.merge(left=PROPERTY_SUBSET, right=data, left_on="smiles", right_on="smiles")
    NEW_DATA.to_csv(NEW_PATH, index=False)

    print("Data persisted!\n")

    return


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--chunk_size", default=10, type=int)
    main(args)
