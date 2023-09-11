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
from chemcaption.molecules import SMILESMolecule
import numpy as np


def test_molecular_formula_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = MolecularFormulaFeaturizer()
    results = featurizer.featurize(molecule)
    assert results[0] == "H2O"

    molecule = SMILESMolecule("C1=CC=CC=C1")
    featurizer = MolecularFormulaFeaturizer()
    results = featurizer.featurize(molecule)
    assert results[0] == "C6H6"

    assert len(featurizer.feature_labels()) == 1

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the molecular formula of the molecule with SMILES c1ccccc1?"  # kekule form
    )
    assert text.to_dict()["filled_completion"] == "Answer: C6H6"


def test_molecular_mass_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = MolecularMassFeaturizer()
    results = featurizer.featurize(molecule)
    assert np.isclose(results[0], 18.015)

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the molecular mass of the molecule with SMILES O?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 18.015"

    assert len(featurizer.feature_labels()) == 1


def test_monoisotopicmass_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = MonoisotopicMolecularMassFeaturizer()
    results = featurizer.featurize(molecule)
    assert np.isclose(results[0], 18.0106)

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the monoisotopic molecular mass of the molecule with SMILES O?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 18.0106"


def test_element_mass_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = ElementMassFeaturizer()
    results = featurizer.featurize(molecule)
    assert len(results[0]) == len(featurizer.feature_labels())
    # ["Carbon", "Hydrogen", "Nitrogen", "Oxygen"] is default preset
    assert results[0][0] == 0
    assert results[0][1] == 2.016
    assert results[0][2] == 0
    assert results[0][3] == 15.999

    text = featurizer.text_featurize(molecule)

    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the total mass of Carbon, Hydrogen, Nitrogen, and Oxygen of the molecule with SMILES O?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 0.0, 2.016, 0.0 and 15.999"


def test_element_mass_proportion_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = ElementMassProportionFeaturizer()

    results = featurizer.featurize(molecule)
    assert len(results[0]) == len(featurizer.feature_labels())

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the mass proportion of Carbon, Hydrogen, Nitrogen, and Oxygen of the molecule with SMILES O?"
    )


def test_element_count_proportion_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = ElementCountProportionFeaturizer()

    results = featurizer.featurize(molecule)
    assert len(results[0]) == len(featurizer.feature_labels())

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the relative atom count of Carbon, Hydrogen, Nitrogen, and Oxygen of the molecule with SMILES O?"
    )
