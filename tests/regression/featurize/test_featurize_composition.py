# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.composition."""

import numpy as np
import pytest

from chemcaption.featurize.composition import (
    AtomCountFeaturizer,
    DegreeOfUnsaturationFeaturizer,
    ElementCountFeaturizer,
    ElementCountProportionFeaturizer,
    ElementMassFeaturizer,
    ElementMassProportionFeaturizer,
    MolecularFormulaFeaturizer,
    MolecularMassFeaturizer,
    MonoisotopicMolecularMassFeaturizer,
)
from chemcaption.molecules import InChIMolecule, SMILESMolecule
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Element mass-related presets
PRESET = ["carbon", "hydrogen", "oxygen", "nitrogen", "phosphorus"]

# Implemented tests for composition-related featurizers.

__all__ = [
    "test_molecular_formula_featurizer",
    "test_molar_mass_featurizer",
    "test_element_mass_featurizer",
    "test_element_mass_proportion_featurizer",
    "test_element_atom_count_featurizer",
    "test_element_atom_count_proportion_featurizer",
    "test_monoisotopic_molar_mass_featurizer",
    "test_atom_count_featurizer",
    "test_get_degree_of_unsaturation_for_mol",
    "test_degree_of_unsaturation_featurizer",
]


"""Test for molecular formular featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property="molecular_formular",
    ),
)
def test_molecular_formula_featurizer(test_input, expected):
    """Test MolecularFormularFeaturizer."""
    featurizer = MolecularFormulaFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


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

    assert np.isclose(results, expected, atol=1.1).all()


"""Test for monoisotopic molecular mass featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property="monoisotopic_molecular_mass",
    ),
)
def test_monoisotopic_molar_mass_featurizer(test_input, expected):
    """Test MonoisotopicMolecularMassFeaturizer."""
    featurizer = MonoisotopicMolecularMassFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected, atol=1.1).all()


"""Test for element mass contribution featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(map(lambda x: x + "_mass", PRESET)),
    ),
)
def test_element_mass_featurizer(test_input, expected):
    """Test ElementMassFeaturizer."""
    featurizer = ElementMassFeaturizer(preset=PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected).all()


"""Test for element mass ratio of contribution featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(map(lambda x: x + "_mass_ratio", PRESET)),
    ),
)
def test_element_mass_proportion_featurizer(test_input, expected):
    """Test ElementMassProportionFeaturizer."""
    featurizer = ElementMassProportionFeaturizer(preset=PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected).all()


"""Test for element atom count contribution featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(map(lambda x: "num_" + x + "_atoms", PRESET)),
    ),
)
def test_element_atom_count_featurizer(test_input, expected):
    """Test ElementCountFeaturizer."""
    featurizer = ElementCountFeaturizer(preset=PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for element atom count ratio contribution featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property=list(map(lambda x: x + "_atom_ratio", PRESET)),
    ),
)
def test_element_atom_count_proportion_featurizer(test_input, expected):
    """Test ElementCountProportionFeaturizer."""
    featurizer = ElementCountProportionFeaturizer(preset=PRESET)
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.isclose(results, expected).all()


"""Test for molecular atom count featurizer."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK,
        representation_name=KIND,
        property="num_atoms",
    ),
)
def test_atom_count_featurizer(test_input, expected):
    """Test AtomCountFeaturizer."""
    featurizer = AtomCountFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected).all()


"""Test for unsaturation degree featurizer."""


def test_get_degree_of_unsaturation_for_mol():
    """Make sure the degree of unsaturation is calculated correctly.

    See some examples here
    https://www.masterorganicchemistry.com/2016/08/26/degrees-of-unsaturation-index-of-hydrogen-deficiency/
    """
    featurizer = DegreeOfUnsaturationFeaturizer()
    mol = SMILESMolecule("c1ccccc1")
    assert featurizer._get_degree_of_unsaturation_for_mol(mol) == 4

    # # dewar benzene
    mol = InChIMolecule("InChI=1S/C6H6/c1-2-6-4-3-5(1)6/h1-6H")
    assert featurizer._get_degree_of_unsaturation_for_mol(mol) == 4

    # hexane
    mol = SMILESMolecule("CCCCCC")
    assert featurizer._get_degree_of_unsaturation_for_mol(mol) == 0

    # cocaine
    mol = SMILESMolecule("CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC")
    assert featurizer._get_degree_of_unsaturation_for_mol(mol) == 8

    # thc
    mol = SMILESMolecule("CCCCCc1cc(c2c(c1)OC([C@H]3[C@H]2C=C(CC3)C)(C)C)O")
    assert featurizer._get_degree_of_unsaturation_for_mol(mol) == 7


def test_degree_of_unsaturation_featurizer():
    """Test degree of unsaturation featurizer."""
    featurizer = DegreeOfUnsaturationFeaturizer()
    results = featurizer.featurize(SMILESMolecule(representation_string="c1ccccc1"))
    assert results == np.array([4.0])

    results = featurizer.featurize(
        SMILESMolecule(representation_string="CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC")
    )
    assert results == np.array([8.0])
