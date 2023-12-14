# -*- coding: utf-8 -*-

"""Script for generating test data."""

import os

import pandas as pd
from rdkit.Chem import Lipinski, rdMolDescriptors

from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.featurize.bonds import BondTypeCountFeaturizer, BondTypeProportionFeaturizer
from chemcaption.featurize.composition import (
    AtomCountFeaturizer,
    ElementCountFeaturizer,
    ElementCountProportionFeaturizer,
    ElementMassFeaturizer,
    ElementMassProportionFeaturizer,
)
from chemcaption.featurize.electronicity import ValenceElectronCountFeaturizer
from chemcaption.featurize.rules import (
    GhoseFilterFeaturizer,
    LeadLikenessFilterFeaturizer,
    LipinskiFilterFeaturizer,
)
from chemcaption.featurize.spatial import (
    AsphericityFeaturizer,
    EccentricityFeaturizer,
    InertialShapeFactorFeaturizer,
    NPRFeaturizer,
    PMIFeaturizer,
)
from chemcaption.featurize.substructure import (
    FragmentSearchFeaturizer,
    IsomorphismFeaturizer,
    TopologyCountFeaturizer,
)
from chemcaption.featurize.stereochemistry import ChiralCenterCountFeaturizer
from chemcaption.featurize.symmetry import RotationalSymmetryNumberFeaturizer, PointGroupFeaturizer
from chemcaption.molecules import SMILESMolecule

BASE_DIR = os.getcwd()

MOLECULAR_BANK = pd.read_json(os.path.join(BASE_DIR, "molecular_bank.json"), orient="index")
PROPERTY_BANK = pd.read_csv(os.path.join(BASE_DIR, "pubchem_response.csv"))

smiles_list = PROPERTY_BANK["smiles"]


def generate_info(string: str):
    """
    Return generated profile for a SMILES string.

    Args:
        string (str): SMILES string.

    Returns:
        (Dict[str, Union[int, float]]): Hash map from property name to property value of type int or float.
    """
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
    lipinski_featurizer = LipinskiFilterFeaturizer()
    ghose_featurizer = GhoseFilterFeaturizer()
    lead_featurizer = LeadLikenessFilterFeaturizer()

    valence_featurizer = ValenceElectronCountFeaturizer()
    isomorphism_featurizer = IsomorphismFeaturizer()

    atom_count_featurizer = AtomCountFeaturizer()
    bond_type_count_featurizer = BondTypeCountFeaturizer()
    bond_type_proportion_featurizer = BondTypeProportionFeaturizer()

    topology_featurizer = TopologyCountFeaturizer.from_preset(preset="organic")

    chiral_featurizer = ChiralCenterCountFeaturizer()
    rotation_symmetry_featurizer = RotationalSymmetryNumberFeaturizer()
    point_group_featurizer = PointGroupFeaturizer()

    # shape_featurizer = MultipleFeaturizer(
    #     featurizers=[
    #         AsphericityFeaturizer(),
    #         EccentricityFeaturizer(),
    #         InertialShapeFactorFeaturizer(),
    #     ]
    # )
    #
    # npr_pmi_featurizer = MultipleFeaturizer(
    #     featurizers = [
    #         NPRFeaturizer(variant="all"),
    #         PMIFeaturizer(variant="all")
    #     ]
    # )

    mol = SMILESMolecule(string)

    wl_hash = isomorphism_featurizer.featurize(mol).item()

    atom_count = atom_count_featurizer.featurize(mol).item()

    num_bonds = len(mol.reveal_hydrogens().GetBonds())

    rotable_strict = rdMolDescriptors.CalcNumRotatableBonds(mol.reveal_hydrogens(), strict=True)
    rotable_non_strict = rdMolDescriptors.CalcNumRotatableBonds(
        mol.reveal_hydrogens(), strict=False
    )

    non_rotable_strict = num_bonds - rotable_strict
    non_rotable_non_strict = num_bonds - rotable_non_strict

    num_donors = Lipinski.NumHDonors(mol.reveal_hydrogens())
    num_acceptors = Lipinski.NumHAcceptors(mol.reveal_hydrogens())

    num_lipinski_violations = lipinski_featurizer.featurize(mol).item()
    num_ghose_violations = ghose_featurizer.featurize(mol).item()
    num_lead_violations = lead_featurizer.featurize(mol).item()

    keys = ["smiles", "weisfeiler_lehman_hash", "num_atoms"]

    keys += bond_type_count_featurizer.feature_labels()
    keys += bond_type_proportion_featurizer.feature_labels()

    keys += [
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
        "num_ghose_violations",
        "num_lead_likeness_violations",
    ]

    masses = mass_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += mass_featurizer.feature_labels()

    mass_ratios = mass_ratio_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += mass_ratio_featurizer.feature_labels()

    counts = count_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += count_featurizer.feature_labels()

    count_ratios = count_ratio_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()
    keys += count_ratio_featurizer.feature_labels()

    num_environments = topology_featurizer.featurize(molecule=mol).reshape(-1).tolist()
    keys += topology_featurizer.feature_labels()

    num_chiral_centers = chiral_featurizer.featurize(molecule=mol).reshape(-1).tolist()
    keys += chiral_featurizer.feature_labels()

    rotation_symmetry_number = rotation_symmetry_featurizer.featurize(molecule=mol).reshape(-1).tolist()
    keys += rotation_symmetry_featurizer.feature_labels()

    point_group = point_group_featurizer.featurize(molecule=mol).reshape(-1).tolist()
    keys += point_group_featurizer.feature_labels()

    valence_count = valence_featurizer.featurize(molecule=mol).item()

    bond_type_counts = bond_type_count_featurizer.featurize(molecule=mol).flatten().tolist()
    bond_type_proportions = (
        bond_type_proportion_featurizer.featurize(molecule=mol).flatten().tolist()
    )

    # shape_features = shape_featurizer.featurize(molecule=mol).flatten().tolist()
    # npr_pmi = npr_pmi_featurizer.featurize(molecule=mol).tolist()

    values = [
        string,
        wl_hash,
        atom_count,
    ]

    values += bond_type_counts
    values += bond_type_proportions

    values += [
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
        num_ghose_violations,
        num_lead_violations,
    ]
    values += masses
    values += mass_ratios

    values += counts
    values += count_ratios
    values += num_environments
    values += num_chiral_centers
    values += rotation_symmetry_number
    values += point_group

    # values += shape_features
    # values += npr_pmi
    #
    # keys += shape_featurizer.feature_labels()
    # keys += npr_pmi_featurizer.feature_labels()

    for preset in ["rings", "organic", "heterocyclic", "warheads", "scaffolds", "amino"]:
        for val in [True, False]:
            smarts_featurizer = FragmentSearchFeaturizer.from_preset(count=val, preset=preset)
            smarts_presence = smarts_featurizer.featurize(molecule=mol).reshape((-1,)).tolist()

            keys += smarts_featurizer.feature_labels()
            values += smarts_presence

    return dict(zip(keys, values))


def extend_dataset(dataset, featurizer):
    new_data = [
        [string] + featurizer.featurize(molecule = SMILESMolecule(string)).flatten().tolist()
        for string in smiles_list
    ]
    print("New data generated!")
    new_data = pd.DataFrame(data=new_data, columns = ["smiles"] + featurizer.feature_labels())
    new_data = pd.merge(left=dataset, right=new_data, left_on="smiles", right_on="smiles")
    print("New data merged and persisted!")
    return new_data

if __name__ == "__main__":
    data = [generate_info(string) for string in smiles_list]

    data = pd.DataFrame(data=data)
    # data.to_csv("data/generated_data.csv", index=False)

    PROPERTY_SUBSET = PROPERTY_BANK.drop(
        labels=[col for col in PROPERTY_BANK.columns if col.__contains__("num")], axis=1
    )

    NEW_DATA = pd.merge(left=PROPERTY_SUBSET, right=data, left_on="smiles", right_on="smiles").rename(
        columns={
            "molar_mass": "molecular_mass",
            "exact_mass": "exact_molecular_mass",
            "monoisotopic_mass": "monoisotopic_molecular_mass",
        }
    )

    # featurizer = MultipleFeaturizer(
    #     featurizers = [
    #         # PointGroupFeaturizer(),
    #         RotationalSymmetryNumberFeaturizer()
    #     ]
    # )

    NEW_PATH = os.path.join(BASE_DIR.replace("legacy", ""), "merged_pubchem_response.csv")
    NEW_DATA.to_csv(NEW_PATH, index=False)
    print("New dataset persisted!")

    # OLD_DATA = pd.read_csv(NEW_PATH)
    # print("Old dataset loaded!")
    #
    # NEW_DATA = extend_dataset(dataset = OLD_DATA, featurizer = featurizer)
    # print("New dataset generated!")

    print(NEW_DATA.columns)
