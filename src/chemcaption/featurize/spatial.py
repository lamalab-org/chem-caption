# -*- coding: utf-8 -*-

"""Featurizers for chemical bond-related information."""

from typing import List, Optional

import numpy as np
from rdkit.Chem import Descriptors3D
from rdkit.Chem.AllChem import EmbedMolecule

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import Molecule

# Implemented bond-related featurizers

__all__ = [
    "ThreeDimensionalFeaturizer",
    "EccentricityFeaturizer",
    "AsphericityFeaturizer",
    "InertialShapeFactorFeaturizer",
]


"""Abstract Featurizer for extracting 3D features from molecule."""


class ThreeDimensionalFeaturizer(AbstractFeaturizer):
    """Abstract class for 3-D featurizers."""

    def __init__(self, conformer_id: Optional[int] = -1, use_masses: bool = True, force=True):
        """Instantiate initialization scheme to be inherited."""
        super().__init__()

        self.conformer_id = conformer_id
        self.use_masses = use_masses
        self.force = force

    def featurize(self, molecule: Molecule) -> None:
        """
        Featurize single molecule instance. Extract eccentricity value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            None.
        """
        pass

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Abstract Featurizer for extracting eccentricity property from molecule."""


class EccentricityFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return eccentricity value of a molecule."""

    def __init__(self, conformer_id: Optional[int] = -1, use_masses: bool = True, force=True):
        """Initialize class object.

        Args:
            conformer_id (Optional[int]): Integer identifier for molecule conformation. Defaults to `-1`.
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool):
        """
        super().__init__(conformer_id=conformer_id, use_masses=use_masses, force=force)

        self.label = ["eccentricity"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract eccentricity value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing eccentricity value.
        """
        mol = molecule.reveal_hydrogens()
        _ = EmbedMolecule(mol)

        eccentricity_value = Descriptors3D.Eccentricity(
            mol, confId=self.conformer_id, force=self.force, useAtomicMasses=self.use_masses
        )
        return np.array([eccentricity_value]).reshape(1, -1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Abstract Featurizer for extracting asphericity property from molecule."""


class AsphericityFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return number of asphericity value of a molecule."""

    def __init__(self, conformer_id: Optional[int] = -1, use_masses: bool = True, force=True):
        """Initialize class object.

        Args:
            conformer_id (Optional[int]): Integer identifier for molecule conformation. Defaults to `-1`.
            use_masses (bool): Utilize elemental masses in asphericity calculation. Defaults to `True`.
            force (bool):
        """
        super().__init__(conformer_id=conformer_id, use_masses=use_masses, force=force)

        self.label = ["asphericity"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract asphericity value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing asphericity value.
        """
        mol = molecule.reveal_hydrogens()
        _ = EmbedMolecule(mol)

        asphericity_value = Descriptors3D.Asphericity(
            mol, confId=self.conformer_id, force=self.force, useAtomicMasses=self.use_masses
        )
        return np.array([asphericity_value]).reshape(1, -1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Abstract Featurizer for extracting asphericity property from molecule."""


class InertialShapeFactorFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return inertia shape factor of a molecule."""

    def __init__(self, conformer_id: Optional[int] = -1, use_masses: bool = True, force=True):
        """Initialize class object.

        Args:
            conformer_id (Optional[int]): Integer identifier for molecule conformation. Defaults to `-1`.
            use_masses (bool): Utilize elemental masses in calculation of inertia shape factor. Defaults to `True`.
            force (bool):
        """
        super().__init__(conformer_id=conformer_id, use_masses=use_masses, force=force)

        self.label = ["inertia_shape_factor"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract inertia shape factor for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing inertia shape factor.
        """
        mol = molecule.reveal_hydrogens()
        _ = EmbedMolecule(mol)

        asphericity_value = Descriptors3D.InertialShapeFactor(
            mol, confId=self.conformer_id, force=self.force, useAtomicMasses=self.use_masses
        )
        return np.array([asphericity_value]).reshape(1, -1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
