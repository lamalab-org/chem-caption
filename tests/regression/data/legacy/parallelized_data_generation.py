# -*- coding: utf-8 -*-

"""Script for generating test data."""

import gc
import os
from argparse import ArgumentParser
from typing import List, Optional

import pandas as pd

from chemcaption.featurize.adaptor import (
    HydrogenAcceptorCountAdaptor,
    HydrogenDonorCountAdaptor,
    NonRotableBondCountAdaptor,
    RotableBondCountAdaptor,
    RotableBondDistributionAdaptor,
    StrictNonRotableBondCountAdaptor,
    StrictRotableBondCountAdaptor,
    StrictRotableBondDistributionAdaptor,
)
from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.featurize.bonds import BondTypeCountFeaturizer
from chemcaption.featurize.composition import (
    AtomCountFeaturizer,
    ElementCountFeaturizer,
    ElementCountProportionFeaturizer,
    ElementMassFeaturizer,
    ElementMassProportionFeaturizer,
)
from chemcaption.featurize.electronicity import ValenceElectronCountFeaturizer
from chemcaption.featurize.rules import LipinskiFilterFeaturizer
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
from chemcaption.molecules import SMILESMolecule

BASE_DIR = os.getcwd()

MOLECULAR_BANK = pd.read_json(os.path.join(BASE_DIR, "molecular_bank.json"), orient="index")
PROPERTY_BANK = pd.read_csv(os.path.join(BASE_DIR, "pubchem_response.csv"))

# Implemented functionality

__all__ = ["main", "generate_featurizer", "persist_data"]


def main(args):
    args = args.parse_args()
    persist_data(chunk_size=args.chunk_size, delete=args.delete)
    return


def generate_featurizer(preset: Optional[List[str]] = None) -> MultipleFeaturizer:
    """
    Return MultipleFeaturizer instance.

    Args:
        None.

    Returns:
        (MultipleFeaturizer): MultipleFeaturizer instance.
    """
    if preset is None:
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

    first_set = [
        IsomorphismFeaturizer(),
        AtomCountFeaturizer(),
        ElementMassFeaturizer(preset=preset),
        ElementMassProportionFeaturizer(preset=preset),
        ElementCountFeaturizer(preset=preset),
        ElementCountProportionFeaturizer(preset=preset),
    ]

    # second_set = [
    #     BondTypeCountFeaturizer(bond_type="all"),
    #     BondTypeProportionFeaturizer(bond_type="all"),
    #     RotableBondCountAdaptor(),
    #     NonRotableBondCountAdaptor(),
    #     StrictRotableBondCountAdaptor(),
    #     StrictNonRotableBondCountAdaptor(),
    #     RotableBondDistributionAdaptor(),
    #     StrictRotableBondDistributionAdaptor(),
    # ]

    # third_set = [
    #     HydrogenDonorCountAdaptor(),
    #     HydrogenAcceptorCountAdaptor(),
    #     ValenceElectronCountFeaturizer(),
    #     LipinskiFilterFeaturizer(),
    #     TopologyCountFeaturizer(),
    #     NPRFeaturizer(variant="all"),
    #     PMIFeaturizer(variant="all"),
    # ]
    #
    # fourth_set = [
    #     AsphericityFeaturizer(),
    #     EccentricityFeaturizer(),
    #     InertialShapeFactorFeaturizer(),
    # ]
    #
    # fifth_set = [
    #     FragmentSearchFeaturizer.from_preset(count=True, preset="rings"),
    #     FragmentSearchFeaturizer.from_preset(count=False, preset="rings"),
    #     FragmentSearchFeaturizer.from_preset(count=True, preset="organic"),
    #     FragmentSearchFeaturizer.from_preset(count=False, preset="organic"),
    #     FragmentSearchFeaturizer.from_preset(count=True, preset="heterocyclic"),
    #     FragmentSearchFeaturizer.from_preset(count=False, preset="heterocyclic"),
    # ]
    #
    # sixth_set = [
    #     FragmentSearchFeaturizer.from_preset(count=True, preset="warheads"),
    #     FragmentSearchFeaturizer.from_preset(count=False, preset="warheads"),
    #     FragmentSearchFeaturizer.from_preset(count=True, preset="scaffolds"),
    #     FragmentSearchFeaturizer.from_preset(count=False, preset="scaffolds"),
    #     FragmentSearchFeaturizer.from_preset(count=True, preset="amino"),
    #     FragmentSearchFeaturizer.from_preset(count=False, preset="amino"),
    # ]

    featurizer = MultipleFeaturizer(
        featurizers=[
            MultipleFeaturizer(featurizers=first_set),
            # MultipleFeaturizer(featurizers=second_set),
            # MultipleFeaturizer(featurizers=third_set),
            # MultipleFeaturizer(featurizers=fourth_set),
            # MultipleFeaturizer(featurizers=fifth_set),
            # MultipleFeaturizer(featurizers=sixth_set)
        ]
    )

    gc.collect()

    return featurizer


def persist_data(chunk_size: int = 30, delete: bool = False) -> None:
    """Break data into chunks and persist.

    Args:
        chunk_size (int): Size of chunks.
        delete (bool): Delete previous version of file. Defaults to `False`.

    Returns:
        None.
    """
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

    # Delete previous data version if required
    if delete:
        if os.path.isfile(NEW_PATH):
            os.remove(NEW_PATH)
            print("Previous data bank deleted!")

    if os.path.isfile(NEW_PATH):
        old_data = pd.read_csv(NEW_PATH)
        start_index = len(old_data)
    else:
        start_index = 0

    # End script run if all strings have been processed
    if start_index >= len(smiles_list):
        print(f"All SMILES strings processed!")
        return

    # If there are strings to process...
    print(
        f"Starting from index {start_index} out of {len(smiles_list)}: {start_index}/{len(smiles_list)}..."
    )

    # Arrange SMILES strings into chunks
    chunks = [
        smiles_list[i : i + chunk_size] for i in range(start_index, len(smiles_list), chunk_size)
    ]

    # Obtain MultipleFeaturizer instance
    featurizer = generate_featurizer()

    running_size = start_index  # Keep track of the count of all SMILES strings processed so far

    for chunk in chunks:  # Process string chunks
        chunk = [SMILESMolecule(string) for string in chunk]
        data = featurizer.generate_data(molecules=chunk, metadata=True)

        if os.path.isfile(NEW_PATH):
            old_data = pd.read_csv(NEW_PATH)
            # columns = old_data.columns
            data = pd.concat((old_data, data), axis=0)

        data.to_csv(NEW_PATH, index=False)
        running_size += len(chunk)

        print(f"Persisted {running_size}/{len(smiles_list)} SMILES strings!\n")

        try:
            del data
            del old_data
        except:
            pass

        gc.collect()

    print("All SMILES strings processed!\n")

    data = pd.read_csv(NEW_PATH)
    NEW_DATA = pd.merge(
        left=PROPERTY_SUBSET, right=data, left_on="smiles", right_on="representation_string"
    )
    # Drop duplicates and redundant columns
    NEW_DATA.drop(labels=["representation_system", "representation_string"], axis=1, inplace=True)
    NEW_DATA.drop_duplicates(inplace=True)

    # Save to disk
    NEW_DATA.to_csv(NEW_PATH, index=False)

    print("Data persisted!\n")

    return


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--chunk_size", default=10, type=int)
    args.add_argument("--delete", default=True, type=bool)

    main(args)
