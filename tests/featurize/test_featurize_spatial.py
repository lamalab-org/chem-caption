# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.spatial submodule."""

import numpy as np

from chemcaption.featurize.spatial import (
    AsphericityFeaturizer,
    AtomVolumeFeaturizer,
    EccentricityFeaturizer,
    InertialShapeFactorFeaturizer,
    NPRFeaturizer,
    PMIFeaturizer,
    RadiusOfGyrationFeaturizer,
    SpherocityIndexFeaturizer,
)
from chemcaption.molecules import SMILESMolecule

__all__ = [
    "test_pmi_featurizer",
    "test_asphericity_featurizer",
    "test_eccentricity_featurizer",
    "test_inertial_shape_factor",
    "test_npr_featurizer",
    "test_radius_of_gyration_featurizer",
    "test_spherocity_index_featurizer",
]


def test_atom_volume_featurizer():
    """Test the AtomVolumeFeaturizer."""

    featurizer = AtomVolumeFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(results) > 0
    assert len(results[0]) == 30

    assert len(featurizer.feature_labels) == len(results[0])

    smiles_list = [SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O"), SMILESMolecule("O")]

    results = featurizer.featurize_many(smiles_list)

    assert len(results) == len(smiles_list)


def test_pmi_featurizer():
    """Test PMIFeaturizer."""
    featurizer = PMIFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 272, atol=2)
    assert np.isclose(results[0][1], 546, atol=2)
    assert np.isclose(results[0][2], 793, atol=2)
    assert len(featurizer.feature_labels) == 3

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert text.to_dict()["filled_prompt"] == (
        "Question: What are the first, second, and third principal moments of inertia (PMI) of the molecule with "
        "SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    # ToDo: make the test below less brittle
    assert text.to_dict()["filled_completion"] == "Answer: 272.4289, 546.3806, and 792.5727"

    try:
        featurizer = PMIFeaturizer(420)
        assert False
    except ValueError:
        assert True

    featurizer = PMIFeaturizer(1)
    assert len(featurizer.feature_labels) == 1


def test_asphericity_featurizer():
    """Test AsphericityFeaturizer."""
    featurizer = AsphericityFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 0.3, atol=0.2)
    assert len(featurizer.feature_labels) == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the asphericity of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"][:-3] == "Answer: 0.3"


def test_eccentricity_featurizer():
    """Test EccentricityFeaturizer."""
    featurizer = EccentricityFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 0.9, atol=0.2)
    assert len(featurizer.feature_labels) == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the eccentricity of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"][:-3] == "Answer: 0.9"


def test_inertial_shape_factor():
    """Test InertialShapeFactorFeaturizer."""
    featurizer = InertialShapeFactorFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 0.002, atol=0.01)
    assert len(featurizer.feature_labels) == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the inertial shape factor of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"][:-3] == "Answer: 0.0"


def test_npr_featurizer():
    """Test NPRFeaturizer."""
    featurizer = NPRFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 0.34, atol=0.2)
    assert len(featurizer.feature_labels) == 2

    assert len(featurizer.get_names) > 0

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)

    assert text.to_dict()["filled_prompt"] == (
        "Question: What are the first and second normalized principal moments ratio (NPR) of the molecule "
        "with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 0.3437 and 0.6894"

    try:
        featurizer = NPRFeaturizer(variant=128)
        assert False
    except ValueError:
        assert True


def test_radius_of_gyration_featurizer():
    """Test RadiusOfGyrationFeaturizer."""
    featurizer = RadiusOfGyrationFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule).item()

    assert np.isclose(results, 2.301, atol=0.2)
    assert len(featurizer.feature_labels) == 1

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the radius of gyration of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 2.3015"


def test_spherocity_index_featurizer():
    """Test SpherocityIndexFeaturizer."""
    featurizer = SpherocityIndexFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule).item()

    assert np.isclose(results, 0.0353, atol=0.2)
    assert len(featurizer.feature_labels) == 1

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the spherocity index of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 0.0353"
