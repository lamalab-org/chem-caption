import numpy as np

from chemcaption.featurize.substructure import SMARTSFeaturizer, TopologyCountFeaturizer
from chemcaption.featurize.text import Prompt
from chemcaption.molecules import SMILESMolecule


def test_topology_count_featurizer():
    molecule = SMILESMolecule("C1=CC=CC=C1")
    featurizer = TopologyCountFeaturizer.from_preset("carbon")
    results = featurizer.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(featurizer.feature_labels())
    text = featurizer.text_featurize(molecule)
    assert isinstance(text, Prompt)

    featurizer = TopologyCountFeaturizer.from_preset("organic")
    results = featurizer.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(featurizer.feature_labels())
    assert results[0][0] == 1
    assert results[0][1] == 1
    assert np.sum(results) == 2

    text = featurizer.text_featurize(molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What are the numbers of topologically unique environments of C, H, N, O, P, S, F, Cl, Br, and I of the molecule with SMILES c1ccccc1?"
    )

    assert text.to_dict()["filled_completion"] == "Answer: 1, 1, 0, 0, 0, 0, 0, 0, 0, and 0"


def test_smarts_featurizer():
    molecule = SMILESMolecule("C1=CC=CC=C1")
    featurizer = SMARTSFeaturizer.from_preset("organic")
    results = featurizer.featurize(molecule)

    assert len(results) == 1
    assert len(results[0]) == len(featurizer.feature_labels())

    assert sum(results[0]) == 0

    text = featurizer.text_featurize(molecule)

    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What are the counts of carboxyl, carbonyl, ether, alkanol, thiol, halogen, amine, amide, and ketone in the molecule with SMILES c1ccccc1?\nConstraint: return a list of integers."
    )

    assert text.to_dict()["filled_completion"] == "Answer: 0, 0, 0, 0, 0, 0, 0, 0, and 0"

    featurizer = SMARTSFeaturizer.from_preset("organic", False)
    results = featurizer.featurize(molecule)

    assert len(results) == 1
    assert len(results[0]) == len(featurizer.feature_labels())

    assert sum(results[0]) == 0

    text = featurizer.text_featurize(molecule)

    assert (
        text.to_dict()["filled_prompt"]
        == "Question: Are carboxyl, carbonyl, ether, alkanol, thiol, halogen, amine, amide, and ketone in the molecule with SMILES c1ccccc1?\nConstraint: return a list of 1s and 0s if the pattern is present or not."
        ""
    )

    assert text.to_dict()["filled_completion"] == "Answer: 0, 0, 0, 0, 0, 0, 0, 0, and 0"

    featurizer = SMARTSFeaturizer(["[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1"], ["benzene"], False)
    results = featurizer.featurize(molecule)

    assert len(results) == 1
    assert len(results[0]) == len(featurizer.feature_labels())

    assert sum(results[0]) == 1

    text = featurizer.text_featurize(molecule)

    assert (
        text.to_dict()["filled_prompt"]
        == "Question: Is benzene in the molecule with SMILES c1ccccc1?\nConstraint: return a list of 1s and 0s if the pattern is present or not."
    )

    assert text.to_dict()["filled_completion"] == "Answer: 1"
