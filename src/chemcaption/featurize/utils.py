# -*- coding: utf-8 -*-

"""Utilities for `featurize` module."""

from functools import lru_cache
from typing import Any, List, Tuple

import numpy as np
from pymatgen.core import IMolecule  # use immutable for caching
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from rdkit import Chem

# Implemented helper functions.

__all__ = [
    "join_list_elements",  # Helper function
    "join_list_elements2",  # Helper function
    "_rdkit_to_pymatgen",  # Helper function
    "_pmg_mol_to_pointgroup_analyzer",  # Helper function
    "get_atom_symbols_and_positions",  # Helper function
    "cached_conformer",  # Helper function
    "apply_featurizer",  # Helper function
    "cached_conformer",
]

def join_list_elements(elements: Any) -> str:
    """Join list elements into a string. First elements separated by comma, last element separated by `and`."""
    if len(elements) == 1:
        return str(elements[0])

    return ", ".join([str(e) for e in elements[:-1]]) + ", and " + str(elements[-1])

def join_list_elements2(names: List[str], elements: Any=None) -> str:
    """Join list elements into a string. All elements separated by comma."""
    
    if len(names) != len(elements):
        raise ValueError(
            f"Length mismatch: names has {len(names)} items but elements has {len(elements)} items"
        )

    parts = []

    for name, amount in zip(names, elements):
        
        if amount == 1:
            parts.append(f"{amount} {name}")
        elif amount == 0:
            parts.append(f"no {name}s")
        else:
            parts.append(f"{amount} {name}s")

    if len(parts) == 0:
        return ""
    elif len(parts) == 1:
        return parts[0]
    else:
        return ", ".join(parts[:-1]) + ", and " + parts[-1]

@lru_cache(maxsize=128)
def _rdkit_to_pymatgen(mol):
    from givemeconformer.api import get_conformer

    c = get_conformer(Chem.MolToSmiles(mol))[0]
    m = IMolecule(*get_atom_symbols_and_positions(c))
    return m


@lru_cache(maxsize=128)
def _pmg_mol_to_pointgroup_analyzer(mol):
    analyzer = PointGroupAnalyzer(mol)
    return analyzer


def get_atom_symbols_and_positions(conf: Any) -> Tuple[List, List]:
    """Returns a touple of atom symbols and positions.

    Args:
        conf (list): List of conformers (atoms).

    Returns:
        tuple(lsit, list): tuple of symbols and positions.
    """

    mol = conf.GetOwningMol()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = conf.GetPositions()
    return symbols, positions


@lru_cache(maxsize=None)
def cached_conformer(smiles, kwargs):
    """Returns cached konformer."""

    from givemeconformer.api import _get_conformer

    mol, conformers = _get_conformer(smiles=smiles, **kwargs)
    for conf in conformers.keys():
        mol.AddConformer(mol.GetConformer(conf))
    return mol


def apply_featurizer(featurize_molecule_pair) -> np.array:
    """Apply a featurizer to a molecule instance to give molecular features.

    Args:
        featurize_molecule_pair (Tuple[AbstractFeaturizer, Molecule]): Pair of:
            (AbstractFeaturizer): Featurizer instance.
            (Molecule): Molecular instance.

    Returns:
        np.array: Featurizer outputs.
    """
    featurizer, molecule = featurize_molecule_pair[0], featurize_molecule_pair[1]
    return (
        featurizer.featurize_many(molecules=molecule)
        if isinstance(molecule, list)
        else featurizer.featurize(molecule=molecule)
    )
