"""Featurizers describing the composition of a molecule."""
from collections import Counter
from functools import lru_cache
from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from chemcaption.featurizers import AbstractFeaturizer
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

__all__ = ["DegreeOfUnsaturationFeaturizer"]


def get_degree_of_unsaturation_for_mol(mol):
    """Returns the degree of unsaturation for a molecule.

    .. math::
        {\displaystyle \mathrm {DU} =1+{\tfrac {1}{2}}\sum n_{i}(v_{i}-2)}

    where ni is the number of atoms with valence vi.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule instance.

    Returns:
        int: Degree of unsaturation.
    """
    # add hydrogens
    mol = Chem.AddHs(mol)
    valence_counter = Counter()
    for atom in mol.GetAtoms():
        valence_counter[atom.GetExplicitValence()] += 1
    du = 1 + 0.5 * sum([n * (v - 2) for v, n in valence_counter.items()])
    return du


class DegreeOfUnsaturationFeaturizer(AbstractFeaturizer):
    """Return the degree of unsaturation."""

    def __init__(self):
        self.template = (
            "What is the degree of unsaturation of a molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "degree of unsaturation",
            }
        ]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance.
        Returns the degree of unsaturation of a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: degree of unsaturation.
        """
        mol = molecule.rdkit_mol
        return np.array([get_degree_of_unsaturation_for_mol(mol)])

    def feature_labels(self) -> List[str]:
        return ["degree_of_unsaturation"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Kevin Maik Jablonka"]
