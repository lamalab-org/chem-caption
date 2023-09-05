# -*- coding: utf-8 -*-

"""Utilities for `featurize` module."""

from typing import List, Tuple, Union, Any
from functools import lru_cache

from givemeconformer.api import get_conformer
from pymatgen.core import IMolecule  # use immutable for caching
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from rdkit import Chem

# Implemented helper functions.

__all__ = [
    "_rdkit_to_pymatgen",  # Helper function
    "_pmg_mol_to_pointgroup_analyzer",  # Helper function
    "get_atom_symbols_and_positions",  # Helper function
    "find_indices", # Helper function
]


@lru_cache(maxsize=128)
def _rdkit_to_pymatgen(mol):
    c = get_conformer(Chem.MolToSmiles(mol))[0]
    m = IMolecule(*get_atom_symbols_and_positions(c))
    return m


@lru_cache(maxsize=128)
def _pmg_mol_to_pointgroup_analyzer(mol):
    analyzer = PointGroupAnalyzer(mol)
    return analyzer


def get_atom_symbols_and_positions(conf):
    mol = conf.GetOwningMol()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = conf.GetPositions()
    return symbols, positions

def find_indices(instance: Any, sequence: Union[List, Tuple]):
    count = sequence.count(instance)
    first_index = 0
    indices = []

    if count > 2:
        for _ in range(count):
            first_index = sequence.index(instance, first_index)
            indices.append(first_index)
    else:
        indices.append(sequence.index(instance))

    return indices