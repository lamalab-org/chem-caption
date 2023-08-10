# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.electronicity subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.electronicity import (
    HydrogenAcceptorCountFeaturizer,
    HydrogenDonorCountFeaturizer,
    ValenceElectronCountFeaturizer,
)

from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for electronicity-related featurizers.

__all__ = [
    "test_num_hydrogen_acceptor_featurizer",
    "test_num_hydrogen_donor_featurizer",
    "test_valence_electron_count_featurizer",
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
def test_num_hydrogen_acceptor_featurizer(test_input, expected):
    """Test HydrogenAcceptorCountFeaturizer."""
    featurizer = HydrogenAcceptorCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for number of Hydrogen bond donors featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_hydrogen_bond_donors"
    ),
)
def test_num_hydrogen_donor_featurizer(test_input, expected):
    """Test HydrogenDonorCountFeaturizer."""
    featurizer = HydrogenDonorCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for number of valence electrons featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_valence_electrons"
    ),
)
def test_valence_electron_count_featurizer(test_input, expected):
    """Test ValenceElectronCountFeaturizer."""
    featurizer = ValenceElectronCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()
