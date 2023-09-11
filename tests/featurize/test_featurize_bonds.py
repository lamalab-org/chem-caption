# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.bonds subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.bonds import (
    BondRotabilityFeaturizer,
    BondTypeCountFeaturizer,
    RotableBondCountFeaturizer,
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


def test_bondrotability_featurizer():
    brf = BondRotabilityFeaturizer()
    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = brf.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(brf.feature_labels())
    assert np.sum(results) == 1
    text = brf.text_featurize(molecule)
    assert isinstance(text, Prompt)


def test_rotable_bond_count_featurizer():
    rbcf = RotableBondCountFeaturizer()
    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = rbcf.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(rbcf.feature_labels())
    assert np.sum(results) == 0
    text = rbcf.text_featurize(molecule)
    assert isinstance(text, Prompt)
