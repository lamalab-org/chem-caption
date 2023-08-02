# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.comparator subpackage."""

import pytest

from chemcaption.featurize.comparator import IsoelectronicComparator
from tests.conftests import DISPATCH_MAP

KIND = "smiles"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for comparator-related featurizers.

__all__ = [
    "test_isoelectronicity_comparator",
]


@pytest.mark.parametrize(
    "test_input, expected",
    [(("N#N", "[C-]#[O+]"), 1), (("[P-3]", "[S-2]"), 1), (("[Al+3]", "[Mg+2]"), 1)],
)
def test_isoelectronicity_comparator(test_input, expected):
    """Test IsoelectronicComparator."""
    smiles1, smiles2 = test_input
    mol1 = MOLECULE(smiles1)
    mol2 = MOLECULE(smiles2)

    comparator = IsoelectronicComparator()

    results = comparator.compare([mol1, mol2])

    assert results == expected
