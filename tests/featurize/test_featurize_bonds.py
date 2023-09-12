# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.bonds subpackage."""

import numpy as np

from chemcaption.featurize.bonds import (
    BondRotabilityFeaturizer,
    BondTypeCountFeaturizer,
    RotableBondCountFeaturizer,
    BondTypeProportionFeaturizer,
)
from chemcaption.molecules import SMILESMolecule
from chemcaption.featurize.text import Prompt


def test_bond_type_featurizer():
    bt = BondTypeCountFeaturizer()
    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = bt.featurize(molecule)
    assert len(results.flatten().tolist()) == len(bt.feature_labels())
    results_dict = dict(zip(bt.feature_labels(), results.flatten().tolist()))
    assert results_dict["num_bonds"] == 12
    assert results_dict["num_aromatic_bonds"] == 6
    assert results_dict["num_single_bonds"] == 6
    text = bt.text_featurize(molecule)
    assert isinstance(text, Prompt)

    assert (
        text.to_dict()["filled_prompt"]
        == """Question: What is the number of unspecified bonds, number of single bonds, number of double bonds, number of triple bonds, number of quadruple bonds, number of quintuble bonds, number of hextuple bonds, number of one-and-a-half bonds, number of two-and-a-half bonds, number of three-and-a-half bonds, number of four-and-a-half bonds, number of five-and-a-half bonds, number of aromatic bonds, number of ionic bonds, number of hydrogen bonds, number of three-center bonds, number of dative one-electron bonds, number of two-electron dative bonds, number of other bonds, number of zero-order bonds, and total number of bonds in the molecule with SMILES c1ccccc1?
Constraint: Return a list of comma separated integers."""
    )

    assert (
        text.to_dict()["filled_completion"]
        == "Answer: 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, and 12"
    )


def test_bondrotability_featurizer():
    brf = BondRotabilityFeaturizer()
    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = brf.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(brf.feature_labels())
    assert np.sum(results) == 1
    text = brf.text_featurize(molecule)
    assert isinstance(text, Prompt)

    assert (
        text.to_dict()["filled_prompt"]
        == """Question: What is the proportion of rotatable and non-rotatable bonds of the molecule with SMILES c1ccccc1?"""
    )

    assert text.to_dict()["filled_completion"] == "Answer: 0.0 and 1.0"


def test_rotable_bond_count_featurizer():
    rbcf = RotableBondCountFeaturizer()
    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = rbcf.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(rbcf.feature_labels())
    assert np.sum(results) == 0
    text = rbcf.text_featurize(molecule)
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

    text = featurizer.text_featurize(molecule)

    assert isinstance(text, Prompt)
    print(text.to_dict()["filled_prompt"])
    assert (
        text.to_dict()["filled_prompt"]
        == """Question: What is the proportion of unspecified bonds, single bonds, double bonds, triple bonds, quadruple bonds, quintuble bonds, hextuple bonds, one-and-a-half bonds, two-and-a-half bonds, three-and-a-half bonds, four-and-a-half bonds, five-and-a-half bonds, aromatic bonds, ionic bonds, hydrogen bonds, three-center bonds, dative one-electron bonds, two-electron dative bonds, other bonds, zero-order bonds, and total number of bonds of the molecule with SMILES c1ccccc1?"""
    )

    assert (
        text.to_dict()["filled_completion"]
        == "Answer: 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0833, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0833, 0.0, and 1.0"
    )
