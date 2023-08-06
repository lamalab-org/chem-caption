# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.bonds subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.bonds import BondRotabilityFeaturizer, RotableBondCountFeaturizer
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for bond-related featurizers.

__all__ = [
    "test_rotable_bond_count_featurizer",
    "test_bond_distribution_featurizer",
]


"""Test for number of rotatable bonds featurizer (strict)."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_rotable_bonds_strict"
    ),
)
def test_rotable_bond_count_featurizer(test_input, expected):
    """Test NumRotableBondsFeaturizer."""
    featurizer = RotableBondCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected.astype(int)).all()


"""Test for number of non-rotatable bonds featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=["rotable_proportion", "non_rotable_proportion"],
    ),
)
def test_bond_distribution_featurizer(test_input, expected):
    """Test BondRotabilityFeaturizer."""
    featurizer = BondRotabilityFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected).all()
