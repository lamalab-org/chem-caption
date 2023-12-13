# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.symmetry subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.symmetry import PointGroupFeaturizer
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for symmetry-related featurizers.

__all__ = [
    "test_point_group_featurizer",
]


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property=["point_group"]
    ),
)
def test_point_group_featurizer(test_input, expected):
    """Test PointGroupFeaturizer."""
    featurizer = PointGroupFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()
