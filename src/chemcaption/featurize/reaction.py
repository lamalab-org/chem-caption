# -*- coding: utf-8 -*-

"""Featurizers for solubility- and reaction-based features."""

import os
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from morfeus import SASA, read_xyz

from chemcaption.featurize.base import MorfeusFeaturizer
from chemcaption.molecules import Molecule

# Implemented reaction- or solubility-based featurizers

__all__ = [
    "SurfaceAccessibleSurfaceAreaFeaturizer",
    "SurfaceAccessibleAtomAreaFeaturizer",
]


class SurfaceAccessibleSurfaceAreaFeaturizer(MorfeusFeaturizer):
    """Return the solvent accessible surface area (SASA) value."""

    def __init__(
            self,
            file_name: Optional[str] = None,
            conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
            morfeus_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Instantiate class.

        Args:
            file_name (Optional[str]): Name for temporary XYZ file.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
            file_name=file_name, conformer_generation_kwargs=conformer_generation_kwargs, morfeus_kwargs=morfeus_kwargs
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

        return SASA(elements, coordinates, **self.morfeus_kwargs)

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


class SurfaceAccessibleAtomAreaFeaturizer(SurfaceAccessibleSurfaceAreaFeaturizer):
    """Return the solvent accessible atom area value."""

    def __init__(
            self,
            file_name: Optional[str] = None,
            conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
            morfeus_kwargs: Optional[Dict[str, Any]] = None,
            atom_index_range: Union[int, Tuple[int, int]] = 100,
    ):
        """Instantiate class.

        Args:
            file_name (Optional[str]): Name for temporary XYZ file.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            atom_index_range (Union[int, List[int, int]]): Range of atoms to calculate areas for.
                Either an integer, or a two-tuple of integers representing lower index and upper index.
        """
        super().__init__(
            file_name=file_name, conformer_generation_kwargs=conformer_generation_kwargs, morfeus_kwargs=morfeus_kwargs
        )

        self._names = [
            {
                "noun": "solvent accessible atom area",
            },
        ]

        self.range = atom_index_range if isinstance(atom_index_range, int) else range(atom_index_range[0], atom_index_range[1])

    def _get_morfeus_instance(self, molecule: Molecule) -> SASA:
        """Return solvent accessible atom area instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (SASA): SASA instance.
        """
        self._mol_to_xyz_file(molecule)  # Persist molecule in XYZ file
        elements, coordinates = read_xyz(self.random_file_name)  # Read file

        os.remove(self.random_file_name)  # Eliminate file

        return SASA(elements, coordinates, **self.morfeus_kwargs)

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing solvent accessible atom area for molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule)
        atom_areas = morfeus_instance.atom_areas
        num_atoms = len(atom_areas)
        atom_areas = [(atom_areas[i] if i <= num_atoms else 0) for i in self.range]
        return np.array(atom_areas).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return [f"solvent_accessible_atom_area_{i}" for i in self.range]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
