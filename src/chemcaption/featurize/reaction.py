# -*- coding: utf-8 -*-

"""Featurizers for solubility- and reaction-based features."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from chemcaption.featurize.base import MorfeusFeaturizer
from chemcaption.molecules import Molecule

# Implemented reaction- or solubility-based featurizers

__all__ = [
    "SolventAccessibleSurfaceAreaFeaturizer",
    "SolventAccessibleVolumeFeaturizer",
    "SolventAccessibleAtomAreaFeaturizer",
]


class SolventAccessibleSurfaceAreaFeaturizer(MorfeusFeaturizer):
    """Return the solvent accessible surface area (SASA) value."""

    def __init__(
        self,
        file_name: Optional[str] = None,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            file_name (Optional[str]): Name for temporary XYZ file.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
            file_name=file_name,
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "solvent accessible surface area",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing solvent accessible surface area (SASA) for molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")
        return np.array([morfeus_instance.area]).reshape(1, -1)

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
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class SolventAccessibleVolumeFeaturizer(MorfeusFeaturizer):
    """Return the solvent accessible volume value for a molecule."""

    def __init__(
        self,
        file_name: Optional[str] = None,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            file_name (Optional[str]): Name for temporary XYZ file.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
            file_name=file_name,
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "solvent accessible volume",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing solvent accessible volume for molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")
        return np.array([morfeus_instance.volume]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["solvent_accessible_volume"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class SolventAccessibleAtomAreaFeaturizer(MorfeusFeaturizer):
    """Return the solvent accessible atom area value for a molecule."""

    def __init__(
        self,
        file_name: Optional[str] = None,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
        atom_indices: Union[int, List[int]] = 100,
        as_range: bool = False,
    ):
        """Instantiate class.

        Args:
            file_name (Optional[str]): Name for temporary XYZ file.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            atom_indices (Union[int, List[int]]): Range of atoms to calculate areas for. Either:
                - an integer,
                - a list of integers, or
                - a two-tuple of integers representing lower index and upper index.
            as_range (bool): Use `atom_indices` parameter as a range of indices or not. Defaults to `False`
        """
        super().__init__(
            file_name=file_name,
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "solvent accessible atom area",
            },
        ]

        self.atom_indices, self.as_range = self._parse_indices(atom_indices, as_range)

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing solvent accessible atom area for atoms in molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="sasa")

        atom_areas = morfeus_instance.atom_areas
        num_atoms = len(atom_areas)

        atom_areas = [(atom_areas[i] if i <= num_atoms else 0) for i in self.atom_indices]

        return np.array(atom_areas).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return [f"solvent_accessible_atom_area_{i}" for i in self.atom_indices]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
