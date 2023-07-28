"""Tests for featurize.composition."""

import numpy as np
from rdkit import Chem

from chemcaption.featurize.composition import (
    DegreeOfUnsaturationFeaturizer,
    get_degree_of_unsaturation_for_mol,
)
from chemcaption.molecules import SMILESMolecule


def test_get_degree_of_unsaturation_for_mol():
    """Make sure the degree of unsaturation is calculated correctly.
    See some examples here
    https://www.masterorganicchemistry.com/2016/08/26/degrees-of-unsaturation-index-of-hydrogen-deficiency/
    """
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert get_degree_of_unsaturation_for_mol(mol) == 4

    # # dewar benzene
    mol = Chem.MolFromInchi("InChI=1S/C6H6/c1-2-6-4-3-5(1)6/h1-6H")
    assert get_degree_of_unsaturation_for_mol(mol) == 4

    # hexane
    mol = Chem.MolFromSmiles("CCCCCC")
    assert get_degree_of_unsaturation_for_mol(mol) == 0

    # cocaine
    mol = Chem.MolFromSmiles("CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC")
    assert get_degree_of_unsaturation_for_mol(mol) == 8

    # thc
    mol = Chem.MolFromSmiles("CCCCCc1cc(c2c(c1)OC([C@H]3[C@H]2C=C(CC3)C)(C)C)O")
    assert get_degree_of_unsaturation_for_mol(mol) == 7


def test_degree_of_unsaturation_featurizer():
    """Test degree of unsaturation featurizer."""
    featurizer = DegreeOfUnsaturationFeaturizer()
    results = featurizer.featurize(SMILESMolecule(representation_string="c1ccccc1"))
    assert results == np.array([4.0])

    results = featurizer.featurize(
        SMILESMolecule(representation_string="CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC")
    )
    assert results == np.array([8.0])
