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
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
        qc_optimize: bool = False,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            qc_optimize (bool): Run QCEngine optimization harness. Defaults to `False`.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
            qc_optimize=qc_optimize,
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
        if self.qc_optimize:
            molecule = self._generate_conformer(molecule=molecule)
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="sasa")
        return np.array([morfeus_instance.area]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
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
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
        qc_optimize: bool = False,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            qc_optimize (bool): Run QCEngine optimization harness. Defaults to `False`.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
            qc_optimize=qc_optimize,
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
        if self.qc_optimize:
            molecule = self._generate_conformer(molecule=molecule)
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="sasa")
        return np.array([morfeus_instance.volume]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
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
    """Return the solvent accessible area value for each atom in a molecule."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
        qc_optimize: bool = False,
        max_index: Optional[int] = None,
        aggregation: Optional[Union[str, List[str]]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            qc_optimize (bool): Run QCEngine optimization harness. Defaults to `False`.
            max_index (Optional[int]): Maximum number of atoms/bonds to consider for feature generation.
            aggregation (Optional[Union[str, List[str]]]): Aggregation to use on generated descriptors.
                Defaults to `None`.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
            qc_optimize=qc_optimize,
            aggregation=aggregation,
        )

        self._names = [
            {
                "noun": "solvent accessible atom area",
            },
        ]

        self.max_index = max_index

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing solvent accessible atom area for atoms in molecule instance.
        """
        if self.qc_optimize:
            molecule = self._generate_conformer(molecule=molecule)

        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="sasa")

        atom_areas = morfeus_instance.atom_areas
        num_atoms = len(atom_areas)

        if self.max_index is None:
            self.max_index = self.fit_on_atom_counts(molecules=molecule)

        atom_areas = [
            (atom_areas[i] if i <= num_atoms else 0) for i in range(1, self.max_index + 1)
        ]

        if self.aggregation is None:
            # Track atom identities
            atomic_numbers = self._track_atom_identity(molecule=molecule, max_index=self.max_index)

            # Combine descriptors with atom identities
            output = atom_areas + atomic_numbers
        else:
            if isinstance(self.aggregation, (list, tuple, set)):
                output = [self.aggregation_func[agg](atom_areas) for agg in self.aggregation]
            else:
                output = self.aggregation_func[self.aggregation](atom_areas)

        return np.array(output).reshape(1, -1)

    def featurize_many(self, molecules: List[Molecule]) -> np.array:
        """
        Featurize a sequence of Molecule objects.

        Args:
            molecules (List[Molecule]): A sequence of molecule representations.

        Returns:
            (np.array): An array of features for each molecule instance.
        """
        self.max_index = self.fit_on_atom_counts(molecules=molecules)

        return super().featurize_many(molecules=molecules)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        if self.aggregation is None:
            return [f"solvent_accessible_atom_area_{i}" for i in range(self.max_index)] + [
                f"atomic_number_{i}" for i in range(self.max_index)
            ]
        else:
            if isinstance(self.aggregation, (list, set, tuple)):
                return [f"solvent_accessible_atom_area_{agg}" for agg in self.aggregation]
            else:
                return ["solvent_accessible_atom_area_" + self.aggregation]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
