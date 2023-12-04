# -*- coding: utf-8 -*-

"""Regression tests for chemcaption.featurize.spatial submodule."""

import numpy as np
import pytest

from chemcaption.featurize.spatial import (
    AsphericityFeaturizer,
    EccentricityFeaturizer,
    InertialShapeFactorFeaturizer,
    NPRFeaturizer,
    PMIFeaturizer,
    RadiusOfGyrationFeaturizer,
    SpherocityIndexFeaturizer,
)
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]


# Implemented tests for spatial-related featurizers.

__all__ = [
    "test_pmi_featurizer",
    "test_asphericity_featurizer",
    "test_eccentricity_featurizer",
    "test_inertial_shape_factor",
    "test_npr_featurizer",
    "test_radius_of_gyration_featurizer",
    "test_spherocity_index_featurizer",
]


"""Test for PMI featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[f"pmi{i}_value" for i in range(1, 4)],
    ),
)
def test_pmi_featurizer(test_input, expected):
    """Test PMIFeaturizer."""
    featurizer = PMIFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for featurizer for asphericity."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property=["asphericity"]
    ),
)
def test_asphericity_featurizer(test_input, expected):
    """Test AsphericityFeaturizer."""
    featurizer = AsphericityFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for featurizer for eccentricity."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property=["eccentricity"]
    ),
)
def test_eccentricity_featurizer(test_input, expected):
    """Test EccentricityFeaturizer."""
    featurizer = EccentricityFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for featurizer for obtaining inertial shape factor."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property=["inertial_shape_factor"]
    ),
)
def test_inertial_shape_factor(test_input, expected):
    """Test InertialShapeFactorFeaturizer."""
    featurizer = InertialShapeFactorFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for featurizer for obtaining NPR."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=[f"npr{i}_value" for i in range(1, 3)],
    ),
)
def test_npr_featurizer(test_input, expected):
    """Test NPRFeaturizer."""
    featurizer = NPRFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for featurizer for obtaining the radius of gyration."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property=["radius_of_gyration"]
    ),
)
def test_radius_of_gyration_featurizer(test_input, expected):
    """Test RadiusOfGyrationFeaturizer."""
    featurizer = RadiusOfGyrationFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for featurizer for obtaining the radius of gyration."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property=["spherocity_index"]
    ),
)
def test_spherocity_index_featurizer(test_input, expected):
    """Test SpherocityIndexFeaturizer."""
    featurizer = SpherocityIndexFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()
