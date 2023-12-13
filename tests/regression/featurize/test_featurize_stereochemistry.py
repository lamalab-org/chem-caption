# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.stereochemistry subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.stereochemistry import ChiralCenterCountFeaturizer
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for stereochemistry-related featurizers.

__all__ = [
    "test_num_chiral_centers",
]


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property=["num_chiral_centers"]
    ),
)
def test_num_chiral_centers(test_input, expected):
    """Test ChiralCenterCountFeaturizer."""
    featurizer = ChiralCenterCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()
