# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.comparator subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.comparator import (
    IsoelectronicComparator,
    IsomerismComparator,
    IsomorphismComparator,
    ValenceElectronCountComparator,
)
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "smiles"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for comparator-related featurizers.

__all__ = [
    "test_isomerism_comparator",
    "test_isomorphism_comparator",
    "test_valence_electron_count_comparator",
    "test_isoelectronicity_comparator",
]


"""Test for valence electron comparator."""


@pytest.mark.parametrize(
    "test_values_1, test_values_2",
    zip(
        extract_molecule_properties(
            property_bank=PROPERTY_BANK, representation_name=KIND, property="num_valence_electrons"
        ),
        sorted(
            extract_molecule_properties(
                property_bank=PROPERTY_BANK,
                representation_name=KIND,
                property="num_valence_electrons",
            )
        ),
    ),
)
def test_valence_electron_count_comparator(test_values_1, test_values_2):
    """Test ValenceElectronCountComparator."""
    test_input_1, expected_1 = test_values_1
    test_input_2, expected_2 = test_values_2

    featurizer = ValenceElectronCountComparator()

    molecule_1 = MOLECULE(test_input_1)
    molecule_2 = MOLECULE(test_input_2)

    results = featurizer.compare([molecule_1, molecule_2])

    assert results == (expected_1.astype(int) == expected_2.astype(int))


"""Test for isomerism comparator."""


@pytest.mark.parametrize(
    "test_values_1, test_values_2",
    zip(
        extract_molecule_properties(
            property_bank=PROPERTY_BANK, representation_name=KIND, property="molecular_formular"
        ),
        sorted(
            extract_molecule_properties(
                property_bank=PROPERTY_BANK, representation_name=KIND, property="molecular_formular"
            )
        ),
    ),
)
def test_isomerism_comparator(test_values_1, test_values_2):
    """Test IsomerismComparator."""
    test_input_1, expected_1 = test_values_1
    test_input_2, expected_2 = test_values_2

    featurizer = IsomerismComparator()

    molecule_1 = MOLECULE(test_input_1)
    molecule_2 = MOLECULE(test_input_2)

    results = featurizer.compare([molecule_1, molecule_2])

    assert np.equal(results, np.equal(expected_1, expected_2).all()).all()


"""Test for isomorphism comparator."""


@pytest.mark.parametrize(
    "test_values_1, test_values_2",
    zip(
        extract_molecule_properties(
            property_bank=PROPERTY_BANK, representation_name=KIND, property="weisfeiler_lehman_hash"
        ),
        sorted(
            extract_molecule_properties(
                property_bank=PROPERTY_BANK,
                representation_name=KIND,
                property="weisfeiler_lehman_hash",
            )
        ),
    ),
)
def test_isomorphism_comparator(test_values_1, test_values_2):
    """Test IsomorphismComparator."""
    test_input_1, expected_1 = test_values_1
    test_input_2, expected_2 = test_values_2

    featurizer = IsomorphismComparator()

    molecule_1 = MOLECULE(test_input_1)
    molecule_2 = MOLECULE(test_input_2)

    results = featurizer.compare([molecule_1, molecule_2])

    assert np.equal(results, np.equal(expected_1, expected_2).all()).all()


"""Test for isoelectronicity comparator."""


@pytest.mark.parametrize(
    "test_values_1, test_values_2",
    zip(
        extract_molecule_properties(
            property_bank=PROPERTY_BANK, representation_name=KIND, property="num_valence_electrons"
        ),
        sorted(
            extract_molecule_properties(
                property_bank=PROPERTY_BANK,
                representation_name=KIND,
                property="num_valence_electrons",
            )
        ),
    ),
)
def test_isoelectronicity_comparator(test_values_1, test_values_2):
    """Test IsoelectronicComparator."""
    test_input_1, expected_1 = test_values_1
    test_input_2, expected_2 = test_values_2

    featurizer = IsoelectronicComparator()

    molecule_1 = MOLECULE(test_input_1)
    molecule_2 = MOLECULE(test_input_2)

    results = featurizer.compare([molecule_1, molecule_2])

    assert np.equal(results, np.equal(expected_1, expected_2).all()).all()
