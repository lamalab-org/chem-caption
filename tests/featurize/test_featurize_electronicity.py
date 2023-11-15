import numpy as np

from chemcaption.featurize.electronicity import (
    HydrogenAcceptorCountFeaturizer,
    HydrogenDonorCountFeaturizer,
    ValenceElectronCountFeaturizer,
)
from chemcaption.molecules import SMILESMolecule


def test_hydrogen_acceptor_count_featurizer():
    """Test HydrogenAcceptorCountFeaturizer."""
    featurizer = HydrogenAcceptorCountFeaturizer()
    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.equal(results, 4).all()
    assert len(featurizer.feature_labels()) == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of hydrogen bond acceptors of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 4"


def test_hydrogen_donor_count_featurizer():
    """Test HydrogenDonorCountFeaturizer."""
    featurizer = HydrogenDonorCountFeaturizer()
    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.equal(results, 1).all()
    assert len(featurizer.feature_labels()) == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of hydrogen bond donors of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 1"


def test_valence_electron_count_featurizer():
    """Test ValenceElectronCountFeaturizer."""
    featurizer = ValenceElectronCountFeaturizer()
    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.equal(results, 56).all()
    assert len(featurizer.feature_labels()) == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of valence electrons of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 56"
