from chemcaption.featurize.adaptor import (
    ValenceElectronCountAdaptor,
    StrictRotableBondDistributionAdaptor,
)

from chemcaption.molecules import SMILESMolecule
import numpy as np


def test_valence_electron_count():
    molecule = SMILESMolecule("C1=CC=CC=C1")
    featurizer = ValenceElectronCountAdaptor()
    results = featurizer.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(featurizer.feature_labels())
    assert np.sum(results) == 30

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of valence electrons of the molecule with SMILES c1ccccc1?"
    )

    assert text.to_dict()["filled_completion"] == "Answer: 30"