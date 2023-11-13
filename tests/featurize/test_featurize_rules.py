# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.rules submodule."""

from chemcaption.featurize.rules import (
    GhoseFilterFeaturizer,
    LeadLikenessFilterFeaturizer,
    LipinskiViolationCountFeaturizer,
)
from chemcaption.molecules import SMILESMolecule

__all__ = [
    "test_lipinski_violation_count_featurizer",
    "test_ghose_filter_featurizer",
    "test_leadlikeness_filter_featurizer",
]


def test_lipinski_violation_count_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = LipinskiViolationCountFeaturizer()

    results = featurizer.featurize(molecule)
    assert results[0] == 0

    molecule = SMILESMolecule("CCCCCCCCCCCCCCCC")

    results = featurizer.featurize(molecule)
    assert results[0] == 1

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of Lipinski violations of the molecule with SMILES CCCCCCCCCCCCCCCC?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 1"


def test_ghose_filter_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = GhoseFilterFeaturizer()

    results = featurizer.featurize(molecule)
    assert results[0] == 0

    molecule = SMILESMolecule("CCCCCCCCCCCCCCCC")
    results = featurizer.featurize(molecule)
    assert results[0] == 3

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of Ghose filter violations of the molecule with SMILES CCCCCCCCCCCCCCCC?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 3"


def test_leadlikeness_filter_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = LeadLikenessFilterFeaturizer()

    results = featurizer.featurize(molecule)
    assert results[0] == 2

    molecule = SMILESMolecule("CCCCCCCCCCCCCCCC")

    results = featurizer.featurize(molecule)
    assert results[0] == 0

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of lead-likeness filter violations of the molecule with SMILES CCCCCCCCCCCCCCCC?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 0"
