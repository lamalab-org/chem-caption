# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.miscellaneous subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.miscellaneous import SVGFeaturizer
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for miscellaneous featurizers.

__all__ = [
    "test_svg_featurizer",
]


"""Test for molecule-to-SVG featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property="svg_string",
    ),
)
def test_svg_featurizer(test_input, expected):
    """Test SVGFeaturizer."""
    featurizer = SVGFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()