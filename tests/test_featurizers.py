# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurizer subpackage."""

import pytest
from rdkit import Chem
from selfies import encoder

from chemcaption.featurizers import (
    HAcceptorCountFeaturizer,
    HDonorCountFeaturizer,
    MolecularMassFeaturizer,
    NumRotableBondsFeaturizer,
)
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "smiles"
MOLECULE = DISPATCH_MAP[KIND]


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(property_bank=PROPERTY_BANK, property="molar_mass"),
)
def test_molar_mass_featurizer(test_input, expected):
    featurizer = MolecularMassFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert round(abs((results - expected).item()), 1) <= 1


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(property_bank=PROPERTY_BANK, property="num_rotable"),
)
def test_num_rotable_bond_featurizer(test_input, expected):
    featurizer = NumRotableBondsFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)

@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(property_bank=PROPERTY_BANK, property="num_hacceptors"),
)
def test_num_hacceptor_featurizer(test_input, expected):
    featurizer = HAcceptorCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)

@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(property_bank=PROPERTY_BANK, property="num_hdonors"),
)
def test_num_hdonor_featurizer(test_input, expected):
    featurizer = HDonorCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)