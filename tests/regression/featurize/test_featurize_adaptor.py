# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.adaptor subpackage."""

import numpy as np
import pytest

from chemcaption.featurize.adaptor import (  # HydrogenAcceptorCountAdaptor,; HydrogenDonorCountAdaptor,; MolecularMassAdaptor,; MonoisotopicMolecularMassAdaptor,; RotableBondCountAdaptor,; StrictRotableBondCountAdaptor,
    ValenceElectronCountAdaptor,
)
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, extract_molecule_properties

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for adaptor-related classes.

__all__ = [
    "test_rdkit_adaptor_valence_electron_count_featurizer",
]


"""Test for number of valence electrons featurizer via higher-level RDKitAdaptor."""


@pytest.mark.parametrize(
    "test_input, expected",
    extract_molecule_properties(
        property_bank=PROPERTY_BANK, representation_name=KIND, property="num_valence_electrons"
    ),
)
def test_rdkit_adaptor_valence_electron_count_featurizer(test_input, expected):
    """Test RDKitAdaptor as ValenceElectronCountFeaturizer (non-strict)."""
    featurizer = ValenceElectronCountAdaptor()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert np.equal(results, expected.astype(int)).all()
