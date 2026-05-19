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
    "join_list_elements",  # Helper function
    "answer_generation",  # Helper function
    "_rdkit_to_pymatgen",  # Helper function
    "_pmg_mol_to_pointgroup_analyzer",  # Helper function
    "get_atom_symbols_and_positions",  # Helper function
    "cached_conformer",  # Helper function
    "apply_featurizer",  # Helper function
    "cached_conformer",
]

#TODO: delete later
def join_list_elements(elements: Any) -> str:
    """Join list elements into a string. First elements separated by comma, last element separated by `and`."""
    if len(elements) == 1:
        return str(elements[0])

    return ", ".join([str(e) for e in elements[:-1]]) + ", and " + str(elements[-1])

def answer_generation(elements: List = None, names: List[str] = None) -> str:
    """Join list elements into a readable string."""

    # VALIDATION PROCESS DEPTH
    
    # 1. parameter None
    
    if names is None:
        if len(elements) == 1:
            return str(elements[0])
        return ", ".join([str(e) for e in elements[:-1]]) + ", and " + str(elements[-1])

    if elements is None:
        raise ValueError("Value Error: No input provided for elements.")
    
    # 2. parameter types - what if not list
    
    if not isinstance(names, list):
        raise TypeError(f"Type Error: Expected 'names' to be a list but got {type(names).__name__} instead.")
    if not isinstance(elements, list):
        raise TypeError(f"Type Error: Expected 'elements' to be a list but got {type(elements).__name__} instead.")

    # 3. validate list element types - what if not string or int
    
    if not all(isinstance(name, str) for name in names):
        raise TypeError("Type Error: All items in 'names' list must be strings.")
    if not all(isinstance(element, int) for element in elements):
        raise TypeError("Type Error: All items in 'elements' list must be integers.")
    
    # 4. validate equal lengths
    
    if len(names) != len(elements):
        raise ValueError(
            f"Length mismatch: names has {len(names)} items but elements has {len(elements) if elements is not None else 0} items"
        )
    
    # 5. Process values

    parts = []

    for name, amount in zip(names, elements):

        # switching bool to int for output
        if isinstance(amount, bool):
            amount = int(amount)

        # missing value handling for elements
        if amount is None:
            if name:
                parts.append(f"unspecified {name}(s)")
            else:
                parts.append("unspecified atom(s)")
            continue

        if name is None:
            if amount == 1:
                parts.append(f"{amount} unspecified atom")
            else:
                parts.append(f"{amount} unspecified atoms")
            continue

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
    elif len(parts) == 2:
        return " and ".join(parts)
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
