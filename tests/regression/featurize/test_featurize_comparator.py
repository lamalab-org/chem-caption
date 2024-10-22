# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.comparator subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.comparator import (
    AtomCountComparator,
    DrugLikenessComparator,
    GhoseFilterComparator,
    IsoelectronicComparator,
    IsomerismComparator,
    IsomorphismComparator,
    LeadLikenessFilterComparator,
    LipinskiFilterComparator,
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
    "test_ghose_filter_comparator",
    "test_drug_likeness_comparator",
    "test_lead_likeness_comparator",
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

    expected = [
        set([s[1][i] for s in string_and_values_pairs])
        for i in range(len(string_and_values_pairs[0]))
    ]
    expected = [(np.array([1]) if len(e) == 1 else np.array([0])) for e in expected]

    expected = np.array([1]) if sum(expected) == len(expected) else np.array([0])

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
    """Test LipinskiFilterComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected = set([s[1][0] for s in string_and_values_pairs])

    expected = np.array([1]) if len(expected) == 1 else np.array([0])

    featurizer = LipinskiFilterComparator()

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


"""Test for Ghose filter comparator."""


@pytest.mark.parametrize(
    "test_values",
    batch_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[
            "num_ghose_violations",
        ],
        batch_size=5,
    ),
)
def test_ghose_filter_comparator(test_values):
    """Test GhoseFilterComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected = set([s[1][0] for s in string_and_values_pairs])

    expected = np.array([1]) if len(expected) == 1 else np.array([0])

    featurizer = GhoseFilterComparator()

    results = featurizer.compare(molecules)

    assert np.equal(results, expected).all()


"""Test for drug-likeness comparator."""


@pytest.mark.parametrize(
    "test_values",
    batch_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[
            "num_lipinski_violations",
            "num_ghose_violations",
            "num_lead_likeness_violations",
        ],
        batch_size=5,
    ),
)
def test_drug_likeness_comparator(test_values):
    """Test DrugLikenessComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected = set([s[1][0] for s in string_and_values_pairs])

    expected = np.array([1]) if len(expected) == 1 else np.array([0])

    featurizer = DrugLikenessComparator()

    results = featurizer.compare(molecules)

    assert np.equal(results, expected).all()


"""Test for lead-likeness comparator."""


@pytest.mark.parametrize(
    "test_values",
    batch_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[
            "num_lead_likeness_violations",
        ],
        batch_size=5,
    ),
)
def test_lead_likeness_comparator(test_values):
    """Test LeadLikenessFilterComparator."""
    string_and_values_pairs = [string_and_values for string_and_values in test_values]
    molecules = [MOLECULE(s[0]) for s in string_and_values_pairs]

    expected = set([s[1][0] for s in string_and_values_pairs])

    expected = np.array([1]) if len(expected) == 1 else np.array([0])

    featurizer = LeadLikenessFilterComparator()

    results = featurizer.compare(molecules)

    assert np.equal(results, expected).all()
