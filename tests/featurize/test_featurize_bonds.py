# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.bonds submodule."""

import numpy as np

from chemcaption.featurize.bonds import (
    BondOrderFeaturizer,
    BondTypeCountFeaturizer,
    BondTypeProportionFeaturizer,
    DipoleMomentsFeaturizer,
    RotableBondCountFeaturizer,
    RotableBondProportionFeaturizer,
    DipoleMomentsFeaturizer,
    BondOrderFeaturizer,
)
from chemcaption.featurize.text import Prompt
from chemcaption.molecules import SMILESMolecule

# Implemented tests for bond-related featurizers.

__all__ = [
    "test_bond_type_count_featurizer",
    "test_rotable_bond_proportion_featurizer",
    "test_rotable_bond_count_featurizer",
    "test_bond_type_proportion_featurizer",
    "test_dipole_moments_featurizer",
    "test_bond_order_featurizer",
]


def test_bond_type_count_featurizer():
    """Tests the BondTypeCountFeaturizer"""

    bt = BondTypeCountFeaturizer()
    assert isinstance(bt.implementors(), list)

    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = bt.featurize(molecule)
    assert len(results.flatten().tolist()) == len(bt.feature_labels)
    results_dict = dict(zip(bt.feature_labels, results.flatten().tolist()))
    assert results_dict["num_bonds"] == 12
    assert results_dict["num_aromatic_bonds"] == 6
    assert results_dict["num_single_bonds"] == 6
    np.random.seed(42)
    text = bt.text_featurize(pos_key="noun", molecule=molecule)
    assert isinstance(text, Prompt)

    assert text.to_dict()["filled_prompt"] == (
        "Question: What are the numbers of the unspecified, single, double, triple, quadruple, "
        "quintuple, hextuple, one-and-a-half, two-and-a-half, three-and-a-half, four-and-a-half, "
        "five-and-a-half, aromatic, ionic, hydrogen, three-center, dative one-electron, dative "
        "two-electron, other, and zero-order bonds of the molecule with SMILES c1ccccc1?\nConstraint: "
        "Return a list of comma separated integers."
    )

    assert (
        text.to_dict()["filled_completion"]
        == "Answer: 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, and 12"
    )

    bt = BondTypeCountFeaturizer(count=False)
    np.random.seed(42)
    text = bt.text_featurize(pos_key="noun", molecule=molecule)
    assert isinstance(text, Prompt)

    assert text.to_dict()["filled_prompt"] == (
        "Question: Are there any unspecified, single, double, triple, quadruple, quintuple, hextuple, "
        "one-and-a-half, two-and-a-half, three-and-a-half, four-and-a-half, five-and-a-half, aromatic, "
        "ionic, hydrogen, three-center, dative one-electron, dative two-electron, other, and zero-order "
        "bond types in the molecule with SMILES c1ccccc1?\nConstraint: Return a list of comma separated "
        "integer / boolean indicators i.e., 0 (or False) for absence, 1 (or True) for presence."
    )

    bt = BondTypeCountFeaturizer(bond_type=["single", "double", "triple"])

    results = bt.featurize(molecule)

    assert len(results[0]) == len(bt.feature_labels)
    assert np.equal(results[0], [1, 0, 0]).all()

    uq = bt._get_unique_bond_types(molecule)

    assert len(uq) == 2
    assert set(uq) == set(["AROMATIC", "SINGLE"])


def test_rotable_bond_proportion_featurizer():
    """Tests the RotableBondProportionFeaturizer"""

    brf = RotableBondProportionFeaturizer()
    assert isinstance(brf.implementors(), list)

    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = brf.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(brf.feature_labels)
    assert np.sum(results) == 1
    text = brf.text_featurize(pos_key="noun", molecule=molecule)
    assert isinstance(text, Prompt)

    assert text.to_dict()["filled_prompt"] == (
        "Question: What are the proportions of rotatable and non-rotatable "
        "bonds of the molecule with SMILES c1ccccc1?"
    )

    assert text.to_dict()["filled_completion"] == "Answer: 0.0 and 1.0"


def test_rotable_bond_count_featurizer():
    """Tests the RotableBondCountFeaturizer"""

    rbcf = RotableBondCountFeaturizer()
    assert isinstance(rbcf.implementors(), list)

    molecule = SMILESMolecule("C1=CC=CC=C1")
    results = rbcf.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(rbcf.feature_labels)
    assert np.sum(results) == 0
    text = rbcf.text_featurize(pos_key="noun", molecule=molecule)
    assert isinstance(text, Prompt)

    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of rotatable bonds of the molecule with SMILES c1ccccc1?"
    )

    assert text.to_dict()["filled_completion"] == "Answer: 0"

    result = rbcf.labeled_featurize(molecule=molecule)

    assert isinstance(result, dict)
    assert len(result.keys()) == len(rbcf.feature_labels)

    rbcf = RotableBondCountFeaturizer()
    text = rbcf.text_featurize(pos_key=None, molecule=molecule)

    assert isinstance(text, Prompt)

    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of rotatable bonds of the molecule with SMILES c1ccccc1?"
    )

    assert text.to_dict()["filled_completion"] == "Answer: 0"

    mols = [SMILESMolecule("C1=CC=CC=C1"), SMILESMolecule("O")]

    try:
        rbcf.text_featurize_many(pos_keys=["hehe"], molecules=mols)
        assert False
    except Exception:
        assert True


def test_bond_type_proportion_featurizer():
    """Tests the BondTypeProportionFeaturizer"""

    featurizer = BondTypeProportionFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("C1=CC=CC=C1")

    results = featurizer.featurize(molecule)

    assert len(results) == 1

    assert len(results[0]) == len(featurizer.feature_labels)

    np.random.seed(42)
    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)

    assert isinstance(text, Prompt)
    assert text.to_dict()["filled_prompt"] == (
        "Question: What are the proportions of the unspecified, single, double, "
        "triple, quadruple, quintuple, hextuple, one-and-a-half, two-and-a-half, "
        "three-and-a-half, four-and-a-half, five-and-a-half, aromatic, ionic, "
        "hydrogen, three-center, dative one-electron, dative two-electron, other, "
        "and zero-order bond types of the molecule with SMILES c1ccccc1?"
        "\nConstraint: Return a list of comma separated floats."
    )
    assert text.to_dict()["filled_completion"] == (
        "Answer: 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, "
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, and 0.0"
    )


def test_dipole_moments_featurizer():
    """Tests the DipoleMomentsFeaturizer"""

    featurizer = DipoleMomentsFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("C1=CC=CC=C1")

    results = featurizer.featurize(molecule)

    assert len(results[0]) == len(featurizer.feature_labels)


def test_bond_order_featurizer():
    """Tests the BondOrderFeaturizer"""

    featurizer = BondOrderFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("C1=CC=CC=C1")

    results = featurizer.featurize(molecule)

    assert len(results[0]) == len(featurizer.feature_labels)

    names = featurizer.feature_names()

    assert isinstance(names, list)
    assert names[0]["noun"] == "bond orders"
