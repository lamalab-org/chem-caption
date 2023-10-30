# -*- coding: utf-8 -*-

"""Test symmetry featurizers."""

import numpy as np

from chemcaption.featurize.symmetry import PointGroupFeaturizer, RotationalSymmetryNumberFeaturizer
from chemcaption.featurize.text import Prompt
from chemcaption.molecules import SMILESMolecule


def test_rotational_symmetry_number():
    """Test rotational symmetry number featurizer."""
    featurizer = RotationalSymmetryNumberFeaturizer()
    molecule = SMILESMolecule(representation_string="CC")
    results = featurizer.featurize(molecule)
    assert results == np.array([6.0])

    molecule = SMILESMolecule(representation_string="CO")
    results = featurizer.featurize(molecule)
    assert results == np.array([1.0])

    text = featurizer.text_featurize(molecule)
    assert isinstance(text, Prompt)
    assert "filled_prompt" in text.__dict__()
    assert (
        text.__dict__()["filled_prompt"]
        == "Question: What is the rotational symmetry number of the molecule with SMILES CO?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 1"


def test_point_group():
    """Test point group featurizer."""
    featurizer = PointGroupFeaturizer()
    molecule = SMILESMolecule(representation_string="CC")
    results = featurizer.featurize(molecule)
    assert results == np.array(["D3d"])

    molecule = SMILESMolecule(representation_string="O=C=O")
    results = featurizer.featurize(molecule)
    assert results == np.array(["D*h"])

    # aspirin
    molecule = SMILESMolecule(representation_string="CC(=O)OC1=CC=CC=C1C(=O)O")
    results = featurizer.featurize(molecule)
    assert results == np.array(["C1"])
    # see https://pqr.pitt.edu/mol/BSYNRYMUTXBXSQ-UHFFFAOYSA-N

    text = featurizer.text_featurize(molecule)
    assert isinstance(text, Prompt)
