# -*- coding: utf-8 -*-

"""Tests for chemcaption.molecules subpackage."""

import pytest

from tests.conftests import (
    DISPATCH_MAP,
    PROPERTY_BANK,
    _convert_molecule,
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
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    return results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(PROPERTY_BANK, in_="smiles", out_="selfies"),
)
def test_smiles_to_selfies(test_input: str, expected: str):
    """Test conversion from SMILES to SELFIES."""
    from_kind, to_kind = "smiles", "selfies"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    return results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(PROPERTY_BANK, in_="smiles", out_="inchi"),
)
def test_smiles_to_inchi(test_input: str, expected: str):
    """Test conversion from SMILES to InChI."""
    from_kind, to_kind = "smiles", "inchi"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    return results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(PROPERTY_BANK, in_="inchi", out_="smiles"),
)
def test_inchi_to_smiles(test_input: str, expected: str):
    """Test conversion from InChI to SMILES."""
    from_kind, to_kind = "inchi", "smiles"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    return results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(PROPERTY_BANK, in_="inchi", out_="selfies"),
)
def test_inchi_to_selfies(test_input: str, expected: str):
    """Test conversion from InChI to SELFIES."""
    from_kind, to_kind = "inchi", "selfies"

    molecule = DISPATCH_MAP[from_kind](
        representation_string=test_input,
    )
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    return results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(PROPERTY_BANK, in_="selfies", out_="inchi"),
)
def test_selfies_to_inchi(test_input: str, expected: str):
    """Test conversion from SELFIES to InChI."""
    from_kind, to_kind = "selfies", "inchi"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    return results == expected
