# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurizer subpackage."""

import numpy as np
import pytest

from chemcaption.featurizers import (
    BondRotabilityFeaturizer,
    ElementCountFeaturizer,
    ElementCountProportionFeaturizer,
    ElementMassFeaturizer,
    ElementMassProportionFeaturizer,
    HAcceptorCountFeaturizer,
    HDonorCountFeaturizer,
    MolecularMassFeaturizer,
    NumRotableBondsFeaturizer,
    SMARTSFeaturizer,
)
from chemcaption.presets import SMARTS_MAP
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Element mass-related presets
PRESET = ["carbon", "hydrogen", "oxygen", "nitrogen", "phosphorus"]

# SMARTS substructure search-related presets
SMARTS_PRESET = "amino"
PRESET_BASE_LABELS = SMARTS_MAP[SMARTS_PRESET]["names"]


"""Test for molecular mass featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="molecular_mass"
    ),
)
def test_molar_mass_featurizer(test_input, expected):
    """Test MolecularMassFeaturizer."""
    featurizer = MolecularMassFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    return np.isclose(results, expected)


"""Test for number of rotatable bonds featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_rotable_bonds_strict"
    ),
)
def test_num_rotable_bond_featurizer(test_input, expected):
    """Test NumRotableBondsFeaturizer."""
    featurizer = NumRotableBondsFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)


"""Test for number of non-rotatable bonds featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=["rotable_proportion", "non_rotable_proportion"],
    ),
)
def test_bond_distribution_featurizer(test_input, expected):
    """Test BondRotabilityFeaturizer."""
    featurizer = BondRotabilityFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    return np.isclose(results, expected)


"""Test for number of Hydrogen bond acceptors featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property="num_hydrogen_bond_acceptors",
    ),
)
def test_num_hacceptor_featurizer(test_input, expected):
    """Test HAcceptorCountFeaturizer."""
    featurizer = HAcceptorCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)


"""Test for number of Hydrogen bond donors featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_hydrogen_bond_donors"
    ),
)
def test_num_hdonor_featurizer(test_input, expected):
    """Test HDonorCountFeaturizer."""
    featurizer = HDonorCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert results == expected.astype(int)


"""Test for element mass contribution featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(map(lambda x: x + "_mass", PRESET)),
    ),
)
def test_mass_featurizer(test_input, expected):
    """Test ElementMassFeaturizer."""
    featurizer = ElementMassFeaturizer(preset=PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    return np.isclose(results, expected)


"""Test for element mass ratio of contribution featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(map(lambda x: x + "_mass_ratio", PRESET)),
    ),
)
def test_mass_proportion_featurizer(test_input, expected):
    """Test ElementMassProportionFeaturizer."""
    featurizer = ElementMassProportionFeaturizer(preset=PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    return np.isclose(results, expected)


"""Test for element atom count contribution featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(map(lambda x: "num_" + x + "_atoms", PRESET)),
    ),
)
def test_atom_count_featurizer(test_input, expected):
    """Test ElementCountFeaturizer."""
    featurizer = ElementCountFeaturizer(preset=PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    return np.isclose(results, expected)


"""Test for element atom count ratio contribution featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(map(lambda x: x + "_atom_ratio", PRESET)),
    ),
)
def test_atom_count_proportion_featurizer(test_input, expected):
    """Test ElementCountProportionFeaturizer."""
    featurizer = ElementCountProportionFeaturizer(preset=PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    return np.isclose(results, expected)


"""Test for SMARTS substructure count featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(
            map(
                lambda x: "".join([("_" if c in "[]()-" else c) for c in x]).lower() + "_count",
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

    return np.equal(results, expected)


"""Test for SMARTS substructure presence featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(
            map(
                lambda x: "".join([("_" if c in "[]()-" else c) for c in x]).lower() + "_presence",
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

    return np.equal(results, expected)
