# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.electronicity subpackage."""

import pytest

from chemcaption.featurize.electronicity import HAcceptorCountFeaturizer, HDonorCountFeaturizer
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for electronicity-related featurizers.

__all__ = [
    "test_num_hacceptor_featurizer",
    "test_num_hdonor_featurizer",
]


"""Test for number of Hydrogen bond acceptors featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property="num_hydrogen_bond_acceptors",
    ),
)
def test_num_hacceptor_featurizer(test_input, expected):
    """Test HAcceptorCountFeaturizer."""
    featurizer = HAcceptorCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)


"""Test for number of Hydrogen bond donors featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_hydrogen_bond_donors"
    ),
)
def test_num_hdonor_featurizer(test_input, expected):
    """Test HDonorCountFeaturizer."""
    featurizer = HDonorCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    return results == expected.astype(int)
