# -*- coding: utf-8 -*-

"""Utilities to export featurizer outputs."""

# Global imports

import json
import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from chemcaption.featurize.adaptor import *
from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.featurize.bonds import *
from chemcaption.featurize.composition import *
from chemcaption.featurize.electronicity import *
from chemcaption.featurize.rules import *
from chemcaption.featurize.stereochemistry import *
from chemcaption.featurize.substructure import *
from chemcaption.featurize.symmetry import *
from chemcaption.molecules import SMILESMolecule
from tests.conftests import FULL_PROPERTY_BANK, PROPERTY_BANK, extract_representation_strings

# Import featurizers



# Implemented functionality

__all__ = ["export_all", "export_gnn_data", "export_llm_data"]

# Collection of featurizer objects

featurizers = [
    RotableBondCountFeaturizer(),
    BondRotabilityFeaturizer(),
    MolecularFormularFeaturizer(),
    MolecularMassFeaturizer(),
    ElementMassFeaturizer(),
    ElementMassProportionFeaturizer(),
    ElementCountFeaturizer(),
    ElementCountProportionFeaturizer(),
    AtomCountFeaturizer(),
    DegreeOfUnsaturationFeaturizer(),
    HydrogenAcceptorCountFeaturizer(),
    HydrogenDonorCountFeaturizer(),
    ValenceElectronCountFeaturizer(),
    LipinskiViolationCountFeaturizer(),
    NumChiralCentersFeaturizer(),
    SMARTSFeaturizer(),
    IsomorphismFeaturizer(),
    TopologyCountFeaturizer(),
    RotationalSymmetryNumberFeaturizer(),
    PointGroupFeaturizer(),
    MonoisotopicMolecularMassAdaptor(),
]


def export_gnn_data(smiles: List[str], featurizer: MultipleFeaturizer):
    """Generate data and export for GNN pretraining.

    Args:
        smiles (List[str]): List of SMILES strings.
        featurizer (MultipleFeaturizer): MultipleFeaturizer instance.

    Returns:
        None.
    """
    print(f"Featurizers are {len(featurizer.featurizers)} in number.\n")

    for f in featurizer.featurizers:
        print(f"{f.__class__.__name__}")

    # Prepare molecules
    mols = [SMILESMolecule(s) for s in smiles]
    new_smiles = np.array(smiles).reshape((-1, 1))

    # Generate features
    features = featurizer.featurize_many(mols)

    # Prepare feature names
    columns = ["SMILES"] + featurizer.feature_labels()

    # Generate dataframe
    data = pd.DataFrame(data=np.concatenate([new_smiles, features], axis=-1), columns=columns)

    # Persist dataframe to local memory
    file_name = os.path.join(os.getcwd(), "data", "gnn", featurizer.__class__.__name__ + ".csv")

    os.makedirs(os.path.join(os.getcwd(), "data", "gnn"), exist_ok=True)

    with open(file_name, "w") as f:
        data.to_csv(f, index=False)

    print("\n\nGNN data export complete!\n")
    return


def export_llm_data(smiles: List[str], featurizer: MultipleFeaturizer):
    """Generate data and export for LLM pretraining.

    Args:
        smiles (List[str]): List of SMILES strings.
        featurizer (MultipleFeaturizer): MultipleFeaturizer instance.

    Returns:
        None.
    """
    print(f"Featurizers are {len(featurizer.featurizers)} in number.\n")

    for f in featurizer.featurizers:
        print(f"{f.__class__.__name__}")

    # Prepare molecules
    mols = [SMILESMolecule(s) for s in smiles]

    # Generate Prompt objects
    prompt_container = featurizer.text_featurize_many(mols)

    # Persist Prompt objects to local memory
    os.makedirs(os.path.join(os.getcwd(), "data", "llm"), exist_ok=True)
    file_name = os.path.join(
        os.path.join(os.getcwd(), "data", "llm"), featurizer.__class__.__name__ + ".jsonl"
    )

    with open(file_name, "w") as f:
        for prompt in tqdm(iter(prompt_container)):
            # for mini_prompt in prompt:
            f.write(json.dumps(prompt) + "\n")

        print("\n\nLLM data export complete!\n")

    return


def export_all(smiles, featurizer):
    """Generate data and export for both GNN and LLM pretraining.

    Args:
        smiles (List[str]): List of SMILES strings.
        featurizer (MultipleFeaturizer): MultipleFeaturizer instance.

    Returns:
        None.
    """
    print(f"Featurizers are {len(featurizer.featurizers)} in number.\n")

    for f in featurizer.featurizers:
        print(f"{f.__class__.__name__}")

    # Export data for GNN pretraining
    export_gnn_data(smiles, featurizer)

    # Export data for LLM pretraining
    export_llm_data(smiles, featurizer)

    return


if __name__ == "__main__":
    smiles = list(
        map(
            lambda x: x[0],
            extract_representation_strings(in_="smiles", molecular_bank=PROPERTY_BANK),
        )
    )
    featurizer = MultipleFeaturizer(featurizers=featurizers)

    # export_all(smiles=smiles, featurizer=featurizer)

    # export_gnn_data(smiles=smiles, featurizer=featurizer)
    export_llm_data(smiles=smiles, featurizer=featurizer)
