"""Featurizers for symmetry."""
from functools import lru_cache
from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from chemcaption.featurizers import AbstractFeaturizer
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule


def _find_chiral_centers(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    chiral_cc = Chem.FindMolChiralCenters(mol)
    return chiral_cc


class NumChiralCentersFeaturizer(AbstractFeaturizer):
    """Return the number of chiral centers."""

    def __init__(self):
        self.template = (
            "What is the number of chiral centers of a molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "number of chiral centers",
            }
        ]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance.
        Returns number of chiral centers in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: number of chiral centers.
        """
        mol = molecule.rdkit_mol
        chiral_cc = _find_chiral_centers(mol)
        return np.array([len(chiral_cc)])

    def feature_labels(self) -> List[str]:
        return ["num_chiral_centers"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Kevin Maik Jablonka"]
