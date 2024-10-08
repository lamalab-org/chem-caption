# -*- coding: utf-8 -*-

"""Featurizers for stereochemistry-related features."""

from typing import Any, List, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import Molecule

__all__ = [
    "ChiralCenterCountFeaturizer",
]

# Implemented stereochemistry-related featurizers


class ChiralCenterCountFeaturizer(AbstractFeaturizer):
    """Return the number of chiral centers."""

    def __init__(self):
        """Instantiate class."""
        super().__init__()

        self._names = [
            {
                "noun": "number of chiral centers",
            }
        ]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of feature labels.
        """
        return ["num_chiral_centers"]

    @staticmethod
    def _find_chiral_centers(molecule: Molecule) -> List[Tuple[Any, Union[Any, str]]]:
        """Return indices for the chiral centers in `molecule`.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            List[Tuple[Any, Union[Any, str]]]: Indices for chiral centers.
        """
        mol = molecule.reveal_hydrogens()
        AllChem.EmbedMolecule(mol)
        Chem.AssignAtomChiralTagsFromStructure(mol)
        chiral_cc = Chem.FindMolChiralCenters(mol)

        return chiral_cc

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Returns number of chiral centers in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: number of chiral centers.
        """
        chiral_cc = self._find_chiral_centers(molecule)
        return np.array([len(chiral_cc)]).reshape((1, 1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Kevin Maik Jablonka"]
