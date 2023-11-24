# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.substructure subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.substructure import (
    FragmentSearchFeaturizer,
    IsomorphismFeaturizer,
    TopologyCountFeaturizer,
)
from chemcaption.presets import SMARTS_MAP
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# SMARTS substructure search-related presets
SMARTS_PRESET = "amino"
PRESET_BASE_LABELS = SMARTS_MAP[SMARTS_PRESET]["names"]

# Topology-related presets
REFERENCE_ATOMIC_NUMBERS = [6, 1, 7, 8, 15, 16, 9, 17, 35, 53]

# Implemented tests for substructure-related featurizers.

__all__ = [
    "test_fragment_count_featurizer",
    "test_fragment_presence_featurizer",
    "test_topology_count_featurizer",
    "test_isomorphism_featurizer",
]


"""Test for SMARTS substructure count featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(
            map(
                lambda x: SMARTS_PRESET
                + "_"
                + "".join([("_" if c in "[]()-" else c) for c in x]).lower()
                + "_count",
                PRESET_BASE_LABELS,
            )
        ),
    ),
)
def test_fragment_count_featurizer(test_input, expected):
    """Test FragmentSearchFeaturizer for SMARTS occurrence count."""
    featurizer = FragmentSearchFeaturizer.from_preset(count=True, preset=SMARTS_PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for SMARTS substructure presence featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(
            map(
                lambda x: SMARTS_PRESET
                + "_"
                + "".join([("_" if c in "[]()-" else c) for c in x]).lower()
                + "_presence",
                PRESET_BASE_LABELS,
            )
        ),
    ),
)
def test_fragment_presence_featurizer(test_input, expected):
    """Test FragmentSearchFeaturizer for SMARTS presence detection."""
    featurizer = FragmentSearchFeaturizer.from_preset(count=False, preset=SMARTS_PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for topology count featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(
            map(
                lambda x: "topology_count_" + str(x),
                REFERENCE_ATOMIC_NUMBERS,
            )
        ),
    ),
)
def test_topology_count_featurizer(test_input, expected):
    """Test TopologyCountFeaturizer for number of unique elemental environments."""
    featurizer = TopologyCountFeaturizer.from_preset(preset="organic")
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for isomorphism featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=["weisfeiler_lehman_hash"],
    ),
)
def test_isomorphism_featurizer(test_input, expected):
    """Test IsomorphismFeaturizer for Weisfeiler-Lehman hash."""
    featurizer = IsomorphismFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()
