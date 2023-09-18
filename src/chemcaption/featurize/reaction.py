# -*- coding: utf-8 -*-

"""Featurizers for solubility- and reaction-based features."""

import os
from typing import Any, Dict, List, Optional

import numpy as np
from morfeus import SASA, read_xyz

from chemcaption.featurize.base import MorfeusFeaturizer
from chemcaption.molecules import Molecule

# Implemented reaction- or solubility-based featurizers

__all__ = [
    "SASAFeaturizer",
]


class SASAFeaturizer(MorfeusFeaturizer):
    """Return the solvent accessible surface area (SASA) value."""

    def __init__(
        self,
        file_name: Optional[str] = None,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            file_name (Optional[str]): Name for temporary XYZ file.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
        """
        super().__init__(
            file_name=file_name, conformer_generation_kwargs=conformer_generation_kwargs
        )

        self._names = [
            {
                "noun": "solvent accessible surface area",
            },
        ]

    def _get_morfeus_instance(self, molecule: Molecule) -> SASA:
        """Return solvent accessible surface area (SASA) instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (SASA): SASA instance.
        """
        self._mol_to_xyz_file(molecule)  # Persist molecule in XYZ file
        elements, coordinates = read_xyz(self.random_file_name)  # Read file

        os.remove(self.random_file_name)  # Eliminate file

        return SASA(elements, coordinates)

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing solvent accessible surface area (SASA) for molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule)
        return morfeus_instance.area.reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["solvent_accessible_solvent_area"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
