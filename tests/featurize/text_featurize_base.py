# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.base submodule."""

from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.featurize.electronicity import HydrogenAcceptorCountFeaturizer
from chemcaption.featurize.stereochemistry import ChiralCenterCountFeaturizer
from chemcaption.molecules import SMILESMolecule


def test_multiple_featurizer():
    smiles = SMILESMolecule("CCCC")

    featurizer = MultipleFeaturizer(
        featurizers=[
            HydrogenAcceptorCountFeaturizer(),
            ChiralCenterCountFeaturizer(),
        ]
    )

    results = featurizer.featurize(smiles)
    assert len(results[0]) == 2
    assert len(results[0]) == len(featurizer.feature_labels())

    text = featurizer.text_featurize(smiles)
    assert len(text) == len(featurizer.featurizers)
