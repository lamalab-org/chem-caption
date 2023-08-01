# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.adaptor subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.adaptor import (
    ExactMolecularMassAdaptor,
    HydrogenAcceptorCountAdaptor,
    HydrogenDonorCountAdaptor,
    MolecularMassAdaptor,
    MonoisotopicMolecularMassAdaptor,
    RotableBondCountAdaptor,
    StrictRotableBondCountAdaptor,
)
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for adaptor-related classes.

__all__ = [
    "test_rdkit_adaptor_molar_mass_featurizer",
    "test_rdkit_adaptor_num_hacceptor_featurizer",
    "test_rdkit_adaptor_num_hdonor_featurizer",
    "test_rdkit_adaptor_strict_num_rotable_bond_featurizer",
    "test_rdkit_adaptor_non_strict_num_rotable_bond_featurizer",
]


"""Test for molecular mass featurizer via higher-level RDKitAdaptor."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="molecular_mass"
    ),
)
def test_rdkit_adaptor_molar_mass_featurizer(test_input, expected):
    """Test RDKitAdaptor as MolecularMassFeaturizer."""
    featurizer = MolecularMassAdaptor()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected, atol=1.1).all()


"""Test for exact molecular mass featurizer via higher-level RDKitAdaptor."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="exact_molecular_mass"
    ),
)
def test_rdkit_adaptor_exact_molar_mass_featurizer(test_input, expected):
    """Test RDKitAdaptor as ExactMolecularMassAdaptor."""
    featurizer = ExactMolecularMassAdaptor()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected, atol=1.1).all()


"""Test for monoisotopic molecular mass featurizer via higher-level RDKitAdaptor."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property="monoisotopic_molecular_mass",
    ),
)
def test_rdkit_adaptor_monoisotopic_molar_mass_featurizer(test_input, expected):
    """Test RDKitAdaptor as MonoisotopicMolecularMassAdaptor."""
    featurizer = MonoisotopicMolecularMassAdaptor()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected, atol=1.1).all()


"""Test for number of Hydrogen bond acceptors via higher-level RDKitAdaptor."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property="num_hydrogen_bond_acceptors",
    ),
)
def test_rdkit_adaptor_num_hacceptor_featurizer(test_input, expected):
    """Test RDKitAdaptor as HAcceptorCountFeaturizer."""
    featurizer = HydrogenAcceptorCountAdaptor()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected.astype(int)).all()


"""Test for number of Hydrogen bond donors via higher-level RDKitAdaptor."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_hydrogen_bond_donors"
    ),
)
def test_rdkit_adaptor_num_hdonor_featurizer(test_input, expected):
    """Test RDKitAdaptor as HDonorCountFeaturizer."""
    featurizer = HydrogenDonorCountAdaptor()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected.astype(int)).all()


"""Test for number of rotatable bonds featurizer (strict) via higher-level RDKitAdaptor."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_rotable_bonds_strict"
    ),
)
def test_rdkit_adaptor_strict_num_rotable_bond_featurizer(test_input, expected):
    """Test RDKitAdaptor as NumRotableBondsFeaturizer (strict)."""
    featurizer = StrictRotableBondCountAdaptor()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected.astype(int)).all()


"""Test for number of rotatable bonds featurizer (non-strict) via higher-level RDKitAdaptor."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_rotable_bonds"
    ),
)
def test_rdkit_adaptor_non_strict_num_rotable_bond_featurizer(test_input, expected):
    """Test RDKitAdaptor as NumRotableBondsFeaturizer (non-strict)."""
    featurizer = RotableBondCountAdaptor()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected.astype(int)).all()
