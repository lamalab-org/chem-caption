# -*- coding: utf-8 -*-

"""Featurizers for symmetry."""

from typing import List

import numpy as np

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.featurize.utils import _pmg_mol_to_pointgroup_analyzer, _rdkit_to_pymatgen
from chemcaption.molecules import Molecule

# Implemented helper functions.

__all__ = [
    "RotationalSymmetryNumber",  # Featurizer
    "PointGroupFeaturizer",  # Featurizer
]


class RotationalSymmetryNumber(AbstractFeaturizer):
    """Obtain the rotational symmetry number of a molecule."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()

        self.template = "What is the {PROPERTY_NAME} for a molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        self._names = [
            {
                "noun": "rotational symmetry number",
            }
        ]
        self.label = ["rotational_symmetry_number"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Returns the rotational symmetry number of a molecule.

        The symmetry number is the number of indistinguishable rotated positions.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (np.array): Rotational symmetry number.
        """
        mol = molecule.rdkit_mol
        m = _rdkit_to_pymatgen(mol)
        analyzer = _pmg_mol_to_pointgroup_analyzer(m)
        return np.array([analyzer.get_rotational_symmetry_number()])

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Kevin Maik Jablonka"]


class PointGroupFeaturizer(AbstractFeaturizer):
    """Return point group of molecule."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()

        self.template = "What is the {PROPERTY_NAME} of a molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        self._names = [
            {
                "noun": "point group",
            }
        ]
        self.label = ["point_group"]

    # ToDo: consider if we want to continue
    # returning the point group as a string
    # I think we have to, because there are infinitely many
    # and for one-hot encoding we would need to know the
    # possible point groups beforehand
    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Returns the point group of a molecule.

        Note that infinity is represented as * in the output.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (np.array): Schoenflies symbol of point group.
        """
        mol = molecule.rdkit_mol
        m = _rdkit_to_pymatgen(mol)
        analyzer = _pmg_mol_to_pointgroup_analyzer(m)
        return np.array([analyzer.get_pointgroup().sch_symbol])

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Kevin Maik Jablonka"]
