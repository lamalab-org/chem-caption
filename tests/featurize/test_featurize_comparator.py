# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.comparator subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.comparator import (
    AtomCountComparator,
    IsoelectronicComparator,
    IsomerismComparator,
    IsomorphismComparator,
    LipinskiViolationCountComparator,
    ValenceElectronCountComparator,
)
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, batch_molecule_properties

KIND = "smiles"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for comparator-related featurizers.

__all__ = [
    "test_isomerism_comparator",
    "test_isomorphism_comparator",
    "test_valence_electron_count_comparator",
    "test_isoelectronicity_comparator",
    "test_lipinski_violation_count_comparator",
    "test_atom_count_comparator",
]


"""Test for valence electron comparator."""


@pytest.mark.parametrize(
    "test_values",
    batch_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[
            "num_valence_electrons",
        ],
        batch_size=5,
    ),
)
def test_valence_electron_count_comparator(test_values):
    """Test ValenceElectronCountComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected = set([s[1][0] for s in string_and_values_pairs])

    expected = np.array([1]) if len(expected) == 1 else np.array([0])

    featurizer = ValenceElectronCountComparator()

    results = featurizer.compare(molecules)

    assert np.equal(results, expected).all()


"""Test for isomerism comparator."""


@pytest.mark.parametrize(
    "test_values",
    batch_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[
            "molecular_formular",
        ],
        batch_size=5,
    ),
)
def test_isomerism_comparator(test_values):
    """Test IsomerismComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected = set([s[1][0] for s in string_and_values_pairs])

    expected = np.array([1]) if len(expected) == 1 else np.array([0])

    featurizer = IsomerismComparator()

    results = featurizer.compare(molecules)

    assert np.equal(results, expected).all()


"""Test for isomorphism comparator."""


@pytest.mark.parametrize(
    "test_values",
    batch_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[
            "weisfeiler_lehman_hash",
        ],
        batch_size=5,
    ),
)
def test_isomorphism_comparator(test_values):
    """Test IsomorphismComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected = set([s[1][0] for s in string_and_values_pairs])

    expected = np.array([1]) if len(expected) == 1 else np.array([0])

    featurizer = IsomorphismComparator()

    results = featurizer.compare(molecules)

    assert np.equal(results, expected).all()


"""Test for isoelectronicity comparator."""


@pytest.mark.parametrize(
    "test_values",
    batch_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=["num_valence_electrons", "weisfeiler_lehman_hash", "num_atoms"],
        batch_size=5,
    ),
)
def test_isoelectronicity_comparator(test_values):
    """Test IsoelectronicComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected_1 = set([s[1][0] for s in string_and_values_pairs])
    expected_2 = set([s[1][1] for s in string_and_values_pairs])
    expected_3 = set([s[1][2] for s in string_and_values_pairs])

    expected_1 = np.array([1]) if len(expected_1) == 1 else np.array([0])
    expected_2 = np.array([1]) if len(expected_2) == 1 else np.array([0])
    expected_3 = np.array([1]) if len(expected_3) == 1 else np.array([0])

    expected = np.array([1]) if (expected_1 & expected_2 & expected_3) else np.array([0])

    featurizer = IsoelectronicComparator()

    results = featurizer.compare(molecules)

    assert np.equal(results, expected).all()


"""Test for valence electron comparator."""


@pytest.mark.parametrize(
    "test_values",
    batch_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[
            "num_lipinski_violations",
        ],
        batch_size=5,
    ),
)
def test_lipinski_violation_count_comparator(test_values):
    """Test LipinskiViolationCountComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected = set([s[1][0] for s in string_and_values_pairs])

    expected = np.array([1]) if len(expected) == 1 else np.array([0])

    featurizer = LipinskiViolationCountComparator()

    results = featurizer.compare(molecules)

    assert np.equal(results, expected).all()


"""Test for valence electron comparator."""


@pytest.mark.parametrize(
    "test_values",
    batch_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[
            "num_atoms",
        ],
        batch_size=5,
    ),
)
def test_atom_count_comparator(test_values):
    """Test AtomCountComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected = set([s[1][0] for s in string_and_values_pairs])

    expected = np.array([1]) if len(expected) == 1 else np.array([0])

    featurizer = AtomCountComparator()

    results = featurizer.compare(molecules)

    assert np.equal(results, expected).all()
