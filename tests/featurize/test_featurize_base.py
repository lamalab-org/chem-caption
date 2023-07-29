# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.substructure subpackage."""

import pytest
import numpy as np

from rdkit.Chem import rdMolDescriptors, Descriptors

from chemcaption.featurize.base import (
    MultipleFeaturizer,
    RDKitAdaptor,
)

from tests.conftests import (
    DISPATCH_MAP,
    PROPERTY_BANK,
    extract_molecule_properties,
)

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for base-implemented classes.

__all__ = [
    "test_rdkit_adaptor_molar_mass_featurizer",
    "test_rdkit_adaptor_num_hacceptor_featurizer",
    "test_rdkit_adaptor_num_hdonor_featurizer",
    "test_rdkit_adaptor_strict_num_rotable_bond_featurizer",
    "test_rdkit_adaptor_non_strict_num_rotable_bond_featurizer"
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
    featurizer = RDKitAdaptor(Descriptors.MolWt, "molecular_mass", **{})
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
    featurizer = RDKitAdaptor(Descriptors.NumHAcceptors, "num_hydrogen_bond_acceptors")
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)


"""Test for number of Hydrogen bond donors via higher-level RDKitAdaptor."""

@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_hydrogen_bond_donors"
    ),
)
def test_rdkit_adaptor_num_hdonor_featurizer(test_input, expected):
    """Test RDKitAdaptor as HDonorCountFeaturizer."""
    featurizer = RDKitAdaptor(
        Descriptors.NumHDonors,
        "num_hydrogen_bond_donors",
    )
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert (results == expected.astype(int))


"""Test for number of rotatable bonds featurizer (strict) via higher-level RDKitAdaptor."""
@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_rotable_bonds_strict"
    ),
)
def test_rdkit_adaptor_strict_num_rotable_bond_featurizer(test_input, expected):
    """Test RDKitAdaptor as NumRotableBondsFeaturizer (strict)."""
    featurizer = RDKitAdaptor(
        rdMolDescriptors.CalcNumRotatableBonds, "num_rotable_bonds_strict", **{"strict": True}
    )
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)


"""Test for number of rotatable bonds featurizer (non-strict) via higher-level RDKitAdaptor."""

@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_rotable_bonds"
    ),
)
def test_rdkit_adaptor_non_strict_num_rotable_bond_featurizer(test_input, expected):
    """Test RDKitAdaptor as NumRotableBondsFeaturizer (non-strict)."""
    featurizer = RDKitAdaptor(
        rdMolDescriptors.CalcNumRotatableBonds, "num_rotable_bonds", **{"strict": False}
    )
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)
