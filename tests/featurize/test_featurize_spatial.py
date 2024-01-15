import numpy as np

from chemcaption.featurize.spatial import (
    AsphericityFeaturizer,
    EccentricityFeaturizer,
    InertialShapeFactorFeaturizer,
    NPRFeaturizer,
    PMIFeaturizer,
)
from chemcaption.molecules import SMILESMolecule

__all__ = [
    "test_pmi_featurizer",
    "test_asphericity_featurizer",
    "test_eccentricity_featurizer",
    "test_inertial_shape_factor",
    "test_npr_featurizer",
]


def test_pmi_featurizer():
    """Test PMIFeaturizer."""
    featurizer = PMIFeaturizer()
    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 272, atol=2)
    assert np.isclose(results[0][1], 546, atol=2)
    assert np.isclose(results[0][2], 793, atol=2)
    assert len(featurizer.feature_labels()) == 3

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What are the first, second, and third principal moments of inertia (PMI) of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    # ToDo: make the test below less brittle
    assert text.to_dict()["filled_completion"] == "Answer: 272.4289, 546.3806, and 792.5727"


def test_asphericity_featurizer():
    """Test AsphericityFeaturizer."""
    featurizer = AsphericityFeaturizer()
    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 0.3, atol=0.2)
    assert len(featurizer.feature_labels()) == 1

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the asphericity of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"][:-3] == "Answer: 0.3"


def test_eccentricity_featurizer():
    """Test EccentricityFeaturizer."""
    featurizer = EccentricityFeaturizer()
    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 0.9, atol=0.2)
    assert len(featurizer.feature_labels()) == 1

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the eccentricity of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"][:-3] == "Answer: 0.9"


def test_inertial_shape_factor():
    """Test InertialShapeFactorFeaturizer."""
    featurizer = InertialShapeFactorFeaturizer()
    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 0.002, atol=0.01)
    assert len(featurizer.feature_labels()) == 1

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the inertial shape factor of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"][:-3] == "Answer: 0.0"


def test_npr_featurizer():
    """Test NPRFeaturizer."""
    featurizer = NPRFeaturizer()
    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.isclose(results[0][0], 0.34, atol=0.2)
    assert len(featurizer.feature_labels()) == 2

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What are the first, and second normalized principal moments ratio (NPR) of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"][:-3] == "Answer: 0.3437 and 0.6"
