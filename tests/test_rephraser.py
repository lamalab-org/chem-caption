# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.rephraser submodule."""

import numpy as np

from chemcaption.molecules import SMILESMolecule
from chemcaption.featurize import MolecularFormulaFeaturizer

from chemcaption.rephraser import SmolLMInstructRePhraser

__all__ = ["test_rephraser"]


def test_rephraser():
    """Tests the rephraser."""
    smiles = [SMILESMolecule("O"), SMILESMolecule("C1=CC=CC=C1")]
    featurizer = MolecularFormulaFeaturizer()

    text = featurizer.text_featurize_many(molecules=smiles)

    model = SmolLMInstructRePhraser(samples=4)
    rephrased = model.rephrase(text)

    assert len(rephrased) == len(smiles)

    assert "H2O" in rephrased[0]["answer"]

    representation = text[1].to_dict()["representation"]

    representation = text[1].to_dict()["representation"]

    assert np.array([representation in q for q in rephrased[1]["alternative"]]).all()
