# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.bonds subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.bonds import (
    _MAP_BOND_TYPE_TO_CLEAN_NAME,
    BondTypeCountFeaturizer,
    BondTypeProportionFeaturizer,
    RotableBondCountFeaturizer,
    RotableBondProportionFeaturizer,
)
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

BOND_TYPES = list(_MAP_BOND_TYPE_TO_CLEAN_NAME.keys())

# Implemented tests for bond-related featurizers.

__all__ = [
    "test_rotable_bond_count_featurizer",
    "test_rotable_bond_proportion_featurizer",
    "test_bond_type_count_featurizer",
    "test_bond_type_proportion_featurizer",
]


"""Test for number of rotatable bonds featurizer (strict)."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_rotable_bonds_strict"
    ),
)
def test_rotable_bond_count_featurizer(test_input, expected):
    """Test RotableBondCountFeaturizer."""
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
def test_rotable_bond_proportion_featurizer(test_input, expected):
    """Test RotableBondProportionFeaturizer."""
    featurizer = RotableBondProportionFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected).all()


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property=BOND_TYPES
    ),
)
def test_bond_type_count_featurizer(test_input, expected):
    """Test for BondTypeCountFeaturizer."""
    bond_type = list(map(lambda x: x.split("_")[1] if len(x.split("_")) == 3 else x, BOND_TYPES))
    featurizer = BondTypeCountFeaturizer(bond_type="all" if "num_bonds" in bond_type else bond_type)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected.astype(int)).all()


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(
            map(
                lambda x: x.split("_")[1] + "_bond_proportion",
                [bt for bt in BOND_TYPES if bt != "num_bonds"],
            )
        ),
    ),
)
def test_bond_type_proportion_featurizer(test_input, expected):
    """Test for BondTypeProportionFeaturizer."""
    bond_type = list(
        map(
            lambda x: x.split("_")[1] if len(x.split("_")) == 3 else x,
            [bt for bt in BOND_TYPES if bt != "num_bonds"],
        )
    )

    featurizer = BondTypeProportionFeaturizer(bond_type=bond_type)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected.astype(float)).all()
