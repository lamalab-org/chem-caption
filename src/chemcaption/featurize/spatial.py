# -*- coding: utf-8 -*-

"""Featurizers for chemical bond-related information."""

from typing import Dict, List, Optional, Union

import numpy as np
from rdkit import Chem
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
    "NPRFeaturizer",
    "PMIFeaturizer",
]


"""Abstract Featurizer for extracting 3D features from molecule."""


class ThreeDimensionalFeaturizer(AbstractFeaturizer):
    """Abstract class for 3-D featurizers."""

    def __init__(self, conformer_id: Optional[int] = -1, use_masses: bool = True, force=True):
        """Instantiate initialization scheme to be inherited.

        Args:
            conformer_id (Optional[int]): Integer identifier for molecule conformation. Defaults to `-1`.
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool):
        """
        super().__init__()

        self.conformer_id = conformer_id
        self.use_masses = use_masses
        self.force = force

        self.FUNCTION_MAP = None

    def _base_rdkit_utility_keys(self) -> List[str]:
        """Returns sorted identifiers for `rdkit` functions in function map.

        Args:
            None.

        Returns:
            (List[str]): List of ordered function keys.
        """
        keys = list(k for k in self.FUNCTION_MAP.keys())
        keys.sort()
        return keys

    def _measure_all(self, *x: Chem.Mol, **y: Dict[str, Union[int, str]]):
        """Return results for all possible variants of 3D featurizer.

        Args:
            *x (Chem.Mol): rdkit Molecule object.
            **y (Union[str, int]): Keyword arguments.
        """
        keys = self._base_rdkit_utility_keys()
        results = [self.FUNCTION_MAP[idx](*x, **y) for idx in keys]
        return results

    def featurize(self, molecule: Molecule) -> None:
        """
        Featurize single molecule instance. Extract 3D feature value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Raises:
            (NotImplementedError): Exception signifying lack of implementation.
        """
        raise NotImplementedError

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


"""Abstract Featurizer for extracting Normalized principal moments ratio (NPR) from molecule."""


class NPRFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return the Normalized principal moments ratio (NPR) value of a molecule."""

    def __init__(
        self,
        variant: Union[int, str] = 1,
        conformer_id: Optional[int] = -1,
        use_masses: bool = True,
        force=True,
    ):
        """Initialize class object.

        Args:
            variant (int): Variant of normalized principal moments ratio (NPR) to calculate.
                May take either value of `1` or `2`. Defaults to `1`.
            conformer_id (Optional[int]): Integer identifier for molecule conformation. Defaults to `-1`.
            use_masses (bool): Utilize elemental masses in calculating the NPR. Defaults to `True`.
            force (bool):
        """
        variant = variant if isinstance(variant, int) else variant.lower()

        super().__init__(conformer_id=conformer_id, use_masses=use_masses, force=force)

        if variant not in list(range(1, 3)) + ["all"]:
            raise ValueError("Argument `variant` must have a value of either `1`, `2`, or `all`.")

        self.variant = variant

        self.FUNCTION_MAP = {
            1: Descriptors3D.NPR1,
            2: Descriptors3D.NPR2,
        }

        self._parse_labels()

    def _parse_labels(self) -> None:
        if self.variant == "all":
            self.label = [f"npr{i}_value" for i in range(1, 3)]
        else:
            self.label = [f"npr{self.variant}_value"]
        return

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract NPR value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing value for NPR.
        """
        mol = molecule.reveal_hydrogens()
        _ = EmbedMolecule(mol)

        npr_function = self.FUNCTION_MAP.get(self.variant, self._measure_all)
        npr_value = npr_function(
            mol, confId=self.conformer_id, force=self.force, useAtomicMasses=self.use_masses
        )
        return np.array([npr_value]).reshape(1, -1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Abstract Featurizer for extracting principal moments of inertia (PMI) from molecule."""


class PMIFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return the normalized principal moments ratio (NPR) value of a molecule."""

    def __init__(
        self,
        variant: Union[int, str] = 1,
        conformer_id: Optional[int] = -1,
        use_masses: bool = True,
        force=True,
    ):
        """Initialize class object.

        Args:
           variant(int): Variant of principal moments of inertia (PMI) to calculate.
                May take either value of `1`, `2`, or `3`. Defaults to `1`.
            conformer_id (Optional[int]): Integer identifier for molecule conformation. Defaults to `-1`.
            use_masses (bool): Utilize elemental masses in calculating the PMI. Defaults to `True`.
            force (bool):
        """

        super().__init__(conformer_id=conformer_id, use_masses=use_masses, force=force)

        variant = variant if isinstance(variant, int) else variant.lower()

        if variant not in list(range(1, 4)) + ["all"]:
            raise ValueError("Argument `pmi` must have a value of either `1`, `2`, `3`, or `all`.")

        self.variant = variant

        self.FUNCTION_MAP = {1: Descriptors3D.PMI1, 2: Descriptors3D.PMI2, 3: Descriptors3D.PMI3}

        self._parse_labels()

    def _parse_labels(self) -> None:
        if self.variant == "all":
            self.label = [f"pmi{i}_value" for i in range(1, 4)]
        else:
            self.label = [f"pmi{self.variant}_value"]
        return

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract PMI value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing value for PMI.
        """
        mol = molecule.reveal_hydrogens()
        _ = EmbedMolecule(mol)

        pmi_function = self.FUNCTION_MAP.get(self.variant, self._measure_all)

        pmi_value = pmi_function(
            mol, confId=self.conformer_id, force=self.force, useAtomicMasses=self.use_masses
        )
        return np.array([pmi_value]).reshape(1, -1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
