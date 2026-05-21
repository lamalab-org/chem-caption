# -*- coding: utf-8 -*-

"""Utilities for `featurize` module."""

from functools import lru_cache
from typing import Any, List, Optional, Tuple

import numpy as np
from pymatgen.core import IMolecule  # use immutable for caching
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from rdkit import Chem

# Implemented helper functions.

__all__ = [
    "answer_generation",  # Helper function
    "_rdkit_to_pymatgen",  # Helper function
    "_pmg_mol_to_pointgroup_analyzer",  # Helper function
    "get_atom_symbols_and_positions",  # Helper function
    "cached_conformer",  # Helper function
    "apply_featurizer",  # Helper function
    "cached_conformer",
]

def _format_element(e):
    """Format a single element to its display string."""
    if isinstance(e, (bool, np.bool_)):
        return str(int(e))           # True -> "1", False -> "0"
    if isinstance(e, (float, np.floating)):
        return f"{float(e):.4f}"
    return str(e)

def _join_readable(parts: List[str]) -> str:
    """Join a list of strings as 'a', 'a and b', or 'a, b, and c'."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    sep = " and "
    return ", ".join(parts[:-1]) + sep + parts[-1]

def answer_generation(elements: Optional[List] = None, names: Optional[List[str]] = None) -> str:
    """Join list elements into a readable string."""

    # VALIDATION PROCESS    
    # 1. elements is None

    if elements is None:
        raise ValueError("Value Error: No input provided for elements.")
    if not isinstance(elements, list):
        raise TypeError(f"Type Error: Expected 'elements' to be a list but got {type(elements).__name__} instead.")

    # 2. names is None
    
    if names is None:
        formatted = [_format_element(e) for e in elements]
        return _join_readable(formatted)
    
    # 3. validate list element types - what if not string or int
    
    if not all(isinstance(element, int) for element in elements):
        raise TypeError("Type Error: All items in 'elements' list must be integers.")
    if not all(isinstance(name, str) for name in names):
        raise TypeError("Type Error: All items in 'names' list must be strings.")
    
    # 4. validate equal lengths
    
    if len(names) != len(elements):
        raise ValueError(
            f"Length mismatch: names has {len(names)} items but elements has {len(elements) if elements is not None else 0} items"
        )
    
    # 5. Process values

    parts: List[str] = []

    for name, amount in zip(names, elements):
        # switching bool to int for output
        if isinstance(amount, bool):
            amount = int(amount)

        if amount == 1:
            parts.append(f"{amount} {name}")
        elif amount == 0:
            parts.append(f"no {name}s")
        else:
            parts.append(f"{amount} {name}s")

    return _join_readable(parts)

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
