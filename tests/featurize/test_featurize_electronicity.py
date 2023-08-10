# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.electronicity subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.electronicity import (
    HydrogenAcceptorCountFeaturizer,
    HydrogenDonorCountFeaturizer,
    ValenceElectronCountFeaturizer,
)

from chemcaption.featurize.comparator import IsoelectronicComparator

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


"""Test for isoelectronicity featurizer."""


@pytest.mark.parametrize(
    "test_values_1, test_values_2",
    zip(
        extract_molecule_properties(
            property_bank=PROPERTY_BANK, representation_name=KIND, property="num_valence_electrons"
        ),
        sorted(
            extract_molecule_properties(
                property_bank=PROPERTY_BANK, representation_name=KIND, property="num_valence_electrons"
            )
        )
    ),
)
def test_isoelectronic_comparator(test_values_1, test_values_2):
    """Test IsoelectronicComparator."""
    test_input_1, expected_1 = test_values_1
    test_input_2, expected_2 = test_values_2

    featurizer = IsoelectronicComparator()

    molecule_1 = MOLECULE(test_input_1)
    molecule_2 = MOLECULE(test_input_2)

    results = featurizer.compare([molecule_1, molecule_2])

    assert results == (expected_1.astype(int) == expected_2.astype(int))
