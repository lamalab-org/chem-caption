# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.bonds submodule."""

import numpy as np

from chemcaption.featurize.bonds import (
    BondTypeCountFeaturizer,
    BondTypeProportionFeaturizer,
    RotableBondCountFeaturizer,
    RotableBondProportionFeaturizer,
)
from chemcaption.featurize.text import Prompt
from chemcaption.molecules import SMILESMolecule

# Implemented tests for bond-related featurizers.

__all__ = [
    "test_bond_type_count_featurizer",
    "test_rotable_bond_proportion_featurizer",
    "test_rotable_bond_count_featurizer",
    "test_bond_type_proportion_featurizer",
]


def test_bond_type_count_featurizer():
    bt = BondTypeCountFeaturizer()
    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = bt.featurize(molecule)
    assert len(results.flatten().tolist()) == len(bt.feature_labels())
    results_dict = dict(zip(bt.feature_labels(), results.flatten().tolist()))
    assert results_dict["num_bonds"] == 12
    assert results_dict["num_aromatic_bonds"] == 6
    assert results_dict["num_single_bonds"] == 6
    text = bt.text_featurize(pos_key="noun", molecule=molecule)
    assert isinstance(text, Prompt)

    assert (
        text.to_dict()["filled_prompt"]
        == """Question: What are the numbers of unspecified, single, double, triple, quadruple, quintuple, hextuple, one-and-a-half, two-and-a-half, three-and-a-half, four-and-a-half, five-and-a-half, aromatic, ionic, hydrogen, three-center, dative one-electron, dative two-electron, other, and zero-order bonds in the molecule with SMILES c1ccccc1?
Constraint: Return a list of comma separated integers."""
    )

    assert (
        text.to_dict()["filled_completion"]
        == "Answer: 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, and 12"
    )

    bt = BondTypeCountFeaturizer(count=False)
    text = bt.text_featurize(pos_key="noun", molecule=molecule)
    assert isinstance(text, Prompt)

    assert (
        text.to_dict()["filled_prompt"]
        == """Question: Are there unspecified, single, double, triple, quadruple, quintuple, hextuple, one-and-a-half, two-and-a-half, three-and-a-half, four-and-a-half, five-and-a-half, aromatic, ionic, hydrogen, three-center, dative one-electron, dative two-electron, other, and zero-order bonds in the molecule with SMILES c1ccccc1?
Constraint: Return a list of comma separated integer boolean indicators (0 for absence, 1 for presence)."""
    )


def test_rotable_bond_proportion_featurizer():
    brf = RotableBondProportionFeaturizer()
    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = brf.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(brf.feature_labels())
    assert np.sum(results) == 1
    text = brf.text_featurize(pos_key="noun", molecule=molecule)
    assert isinstance(text, Prompt)

    assert (
        text.to_dict()["filled_prompt"]
        == """Question: What are the proportions of rotatable and non-rotatable bonds of the molecule with SMILES c1ccccc1?"""
    )

    assert text.to_dict()["filled_completion"] == "Answer: 0.0 and 1.0"


def test_rotable_bond_count_featurizer():
    rbcf = RotableBondCountFeaturizer()
    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = rbcf.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(rbcf.feature_labels())
    assert np.sum(results) == 0
    text = rbcf.text_featurize(pos_key="noun", molecule=molecule)
    assert isinstance(text, Prompt)

    assert (
        text.to_dict()["filled_prompt"]
        == """Question: What is the number of rotatable bonds of the molecule with SMILES c1ccccc1?"""
    )

    assert text.to_dict()["filled_completion"] == "Answer: 0"


def test_bond_type_proportion_featurizer():
    featurizer = BondTypeProportionFeaturizer()

    molecule = SMILESMolecule("C1=CC=CC=C1")

    results = featurizer.featurize(molecule)

    assert len(results) == 1

    assert len(results[0]) == len(featurizer.feature_labels())

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)

    assert isinstance(text, Prompt)
    print(text.to_dict()["filled_prompt"])
    assert (
        text.to_dict()["filled_prompt"]
        == """Question: What are the proportions of unspecified, single, double, triple, quadruple, quintuple, hextuple, one-and-a-half, two-and-a-half, three-and-a-half, four-and-a-half, five-and-a-half, aromatic, ionic, hydrogen, three-center, dative one-electron, dative two-electron, other, and zero-order bonds in the molecule with SMILES c1ccccc1?
Constraint: Return a list of comma separated floats."""
    )
    print(text.to_dict()["filled_completion"])
    assert (
        text.to_dict()["filled_completion"]
        == "Answer: 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, and 0.0"
    )
