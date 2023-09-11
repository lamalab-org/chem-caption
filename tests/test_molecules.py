# -*- coding: utf-8 -*-

"""Tests for chemcaption.molecules subpackage."""

import pytest

from tests.conftests import (
    DISPATCH_MAP,
    PROPERTY_BANK,
    convert_molecule,
    extract_representation_strings,
)


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(PROPERTY_BANK, in_="selfies", out_="smiles"),
)
def test_selfies_to_smiles(test_input: str, expected: str):
    """Test conversion from SELFIES to SMILES."""
    from_kind, to_kind = "selfies", "smiles"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    assert results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(PROPERTY_BANK, in_="smiles", out_="selfies"),
)
def test_smiles_to_selfies(test_input: str, expected: str):
    """Test conversion from SMILES to SELFIES."""
    from_kind, to_kind = "smiles", "selfies"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    assert results == expected


def test_inchi_mol():
    """Test InChIMolecule."""
    from chemcaption.molecules import InChIMolecule

    molecule = InChIMolecule(representation_string="InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H")
    results = molecule.representation_string

    assert results == "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
