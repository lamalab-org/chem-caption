# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.substructure subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.substructure import SMARTSFeaturizer
from chemcaption.presets import SMARTS_MAP
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# SMARTS substructure search-related presets
SMARTS_PRESET = "amino"
PRESET_BASE_LABELS = SMARTS_MAP[SMARTS_PRESET]["names"]

# Implemented tests for substructure-related featurizers.

__all__ = [
    "test_smarts_count_featurizer",
    "test_smarts_presence_featurizer",
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
def test_smarts_count_featurizer(test_input, expected):
    """Test SMARTSFeaturizer for SMARTS occurrence count."""
    featurizer = SMARTSFeaturizer(count=True, names=SMARTS_PRESET)
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
def test_smarts_presence_featurizer(test_input, expected):
    """Test SMARTSFeaturizer for SMARTS presence detection."""
    featurizer = SMARTSFeaturizer(count=False, names=SMARTS_PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()
