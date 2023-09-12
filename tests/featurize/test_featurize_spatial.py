from chemcaption.featurize.spatial import (
    PMIFeaturizer,
    AsphericityFeaturizer,
    EccentricityFeaturizer,
    InertialShapeFactorFeaturizer,
    NPRFeaturizer,
)

from chemcaption.molecules import SMILESMolecule
import numpy as np


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
        == "Question: What is the first, second, and third principal moments of inertia (PMI) of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 272.4289, 546.3806 and 792.5727"
