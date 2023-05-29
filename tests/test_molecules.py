# -*- coding: utf-8 -*-

"""Tests for chemcaption.molecules subpackage."""

import pandas as pd
import pytest
from rdkit import Chem
from selfies import decoder, encoder

from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

data_bank = pd.read_csv("data/molecular_bank.csv", encoding="latin-1")
data_bank["smiles"] = data_bank["smiles"].apply(Chem.CanonSmiles)
molecular_names, molecular_smiles = (
    data_bank["name"].values.tolist(),
    data_bank["smiles"].values.tolist(),
)


molecular_bank = {
    k: {
        "smiles": v,
        "selfies": encoder(v),
        "smiled_selfies": Chem.CanonSmiles(decoder(encoder(v))),
        "inchi": Chem.MolToInchi(Chem.MolFromSmiles(v)),
    }
    for k, v in zip(molecular_names, molecular_smiles)
}

DISPATCH_MAP = {
    "smiles": SMILESMolecule,
    "selfies": SELFIESMolecule,
    "inchi": InChIMolecule,
}


def extract_representation_strings(molecular_bank, in_="smiles", out_="selfies"):
    """Extract molecule representation strings from data bank."""
    in_, out_ = in_.lower(), out_.lower()
    input_output = [(v[in_], v[out_]) for k, v in molecular_bank.items()]
    return input_output


def _convert_molecule(molecule, to_kind="smiles"):
    """Convert molecules between representational systems."""
    to_kind = to_kind.lower()

    if to_kind == "inchi":
        representation_string = Chem.MolToInchi(molecule.rdkit_mol)
    else:
        representation_string = Chem.MolToSmiles(molecule.rdkit_mol)
        representation_string = Chem.CanonSmiles(representation_string)
        if to_kind == "selfies":
            representation_string = encoder(representation_string)

    return DISPATCH_MAP[to_kind](representation_string)


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(molecular_bank, in_="selfies", out_="smiles"),
)
def test_selfies_to_smiles(test_input, expected):
    """Test conversion from SELFIES to SMILES."""
    from_kind, to_kind = "selfies", "smiles"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    assert results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(molecular_bank, in_="smiles", out_="selfies"),
)
def test_smiles_to_selfies(test_input, expected):
    """Test conversion from SMILES to SELFIES."""
    from_kind, to_kind = "smiles", "selfies"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    assert results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(molecular_bank, in_="smiles", out_="inchi"),
)
def test_smiles_to_inchi(test_input, expected):
    """Test conversion from SMILES to InChI."""
    from_kind, to_kind = "smiles", "inchi"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    assert results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(molecular_bank, in_="inchi", out_="smiles"),
)
def test_inchi_to_smiles(test_input, expected):
    """Test conversion from InChI to SMILES."""
    from_kind, to_kind = "inchi", "smiles"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = Chem.CanonSmiles(new_molecule.representation_string)

    assert results == Chem.CanonSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(expected)))


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(molecular_bank, in_="inchi", out_="selfies"),
)
def test_inchi_to_selfies(test_input, expected):
    """Test conversion from InChI to SELFIES."""
    from_kind, to_kind = "inchi", "selfies"

    molecule = DISPATCH_MAP[from_kind](
        representation_string=test_input,
    )
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    assert results == expected


@pytest.mark.parametrize(
    "test_input, expected",
    extract_representation_strings(molecular_bank, in_="selfies", out_="inchi"),
)
def test_selfies_to_inchi(test_input, expected):
    """Test conversion from SELFIES to InChI."""
    from_kind, to_kind = "selfies", "inchi"

    molecule = DISPATCH_MAP[from_kind](representation_string=test_input)
    new_molecule = _convert_molecule(molecule, to_kind=to_kind)
    results = new_molecule.representation_string

    assert results == expected
