# -*- coding: utf-8 -*-

"""Test chemcaption.featurize.rules classes for drug properties."""

import numpy as np
import pytest

from chemcaption.featurize.rules import (
    GhoseFilterFeaturizer,
    LeadLikenessFilterFeaturizer,
    LipinskiFilterFeaturizer,
)
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Element mass-related presets
PRESET = ["carbon", "hydrogen", "oxygen", "nitrogen", "phosphorus"]

# Implemented tests for drug rule-related classes.

__all__ = [
    "test_lipinski_filter_featurizer",
    "test_ghose_filter_featurizer",
    "test_leadlikeness_filter_featurizer",
]


"""Test for Lipinski violation count featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_lipinski_violations"
    ),
)
def test_lipinski_filter_featurizer(test_input, expected):
    """Test LipinskiFilterFeaturizer."""
    featurizer = LipinskiFilterFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for Ghose violation count featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_ghose_violations"
    ),
)
def test_ghose_filter_featurizer(test_input, expected):
    """Test GhoseFilterFeaturizer."""
    featurizer = GhoseFilterFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for lead-likeness violation count featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property="num_lead_likeness_violations",
    ),
)
def test_leadlikeness_filter_featurizer(test_input, expected):
    """Test LeadLikenessFilterFeaturizer."""
    featurizer = LeadLikenessFilterFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()
