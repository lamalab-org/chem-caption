# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurizer subpackage."""

import pytest
from rdkit import Chem
from selfies import encoder

from chemcaption.featurizers import HAcceptorCountFeaturizer
from chemcaption.molecules import InChIMolecule
from tests.conftests import DISPATCH_MAP, MOLECULAR_BANK, extract_representation_strings


def test_num_rotable_bond_featurizer(test_input, expected):
    assert test_input == expected
