# -*- coding: utf-8 -*-

"""Featurizers for 3D (i.e., spatial) information."""

from functools import lru_cache
from typing import Dict, List, Union

import numpy as np
from frozendict import frozendict
from givemeconformer.api import _get_conformer
from rdkit import Chem
from rdkit.Chem import Descriptors3D

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.featurize.utils import join_list_elements
from chemcaption.molecules import Molecule

# Implemented bond-related featurizers

__all__ = [
    "ThreeDimensionalFeaturizer",
    "EccentricityFeaturizer",
    "AsphericityFeaturizer",
    "InertialShapeFactorFeaturizer",
    "NPRFeaturizer",
    "PMIFeaturizer",
    "SpherocityIndexFeaturizer",
    "RadiusOfGyrationFeaturizer",
]


"""Abstract Featurizer for extracting 3D features from molecule."""


@lru_cache(maxsize=None)
def cached_conformer(smiles, kwargs):
    mol, conformers = _get_conformer(smiles=smiles, **kwargs)
    for conf in conformers.keys():
        mol.AddConformer(mol.GetConformer(conf))
    return mol


class ThreeDimensionalFeaturizer(AbstractFeaturizer):
    """Abstract class for 3-D featurizers."""

    def __init__(self, use_masses: bool = True, force=True, conformer_generation_kwargs=None):
        """Instantiate initialization scheme to be inherited.

        Args:
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (dict): Keyword arguments for conformer generation.
        """
        super().__init__()

        self.use_masses = use_masses
        self.force = force

        self.FUNCTION_MAP = None
        self._conf_gen_kwargs = (
            frozendict(conformer_generation_kwargs)
            if conformer_generation_kwargs
            else frozendict({})
        )

    def _get_conformer(self, mol):
        smiles = Chem.MolToSmiles(mol)
        return cached_conformer(smiles, self._conf_gen_kwargs)

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


class EccentricityFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return eccentricity value of a molecule."""

    def __init__(self, use_masses: bool = True, force=True, conformer_generation_kwargs=None):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (dict): Keyword arguments for conformer generation.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "eccentricity"}]

    def feature_labels(self) -> List[str]:
        return ["eccentricity"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract eccentricity value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing eccentricity value.
        """
        mol = molecule.rdkit_mol
        mol = self._get_conformer(mol)

        eccentricity_value = Descriptors3D.Eccentricity(
            mol, force=self.force, useAtomicMasses=self.use_masses
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


class AsphericityFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return number of asphericity value of a molecule."""

    def __init__(self, use_masses: bool = True, force=True, conformer_generation_kwargs=None):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses in asphericity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (dict): Keyword arguments for conformer generation.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "asphericity"}]

    def feature_labels(self) -> List[str]:
        return ["asphericity"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract asphericity value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing asphericity value.
        """
        mol = molecule.reveal_hydrogens()

        mol = self._get_conformer(mol)

        asphericity_value = Descriptors3D.Asphericity(
            mol, force=self.force, useAtomicMasses=self.use_masses
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


class InertialShapeFactorFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return inertial shape factor of a molecule."""

    def __init__(self, use_masses: bool = True, force=True, conformer_generation_kwargs=None):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (dict): Keyword arguments for conformer generation.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "inertial shape factor"}]

    def feature_labels(self) -> List[str]:
        return ["inertial_shape_factor"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract inertia shape factor for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing inertia shape factor.
        """
        mol = molecule.rdkit_mol

        mol = self._get_conformer(mol)

        asphericity_value = Descriptors3D.InertialShapeFactor(
            mol, force=self.force, useAtomicMasses=self.use_masses
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


class NPRFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return the Normalized principal moments ratio (NPR) value of a molecule."""

    def __init__(
        self,
        variant: Union[int, str] = "all",  # today make iterable
        use_masses: bool = True,
        force=True,
        conformer_generation_kwargs=None,
    ):
        """Initialize class object.

        Args:
            variant (int): Variant of normalized principal moments ratio (NPR) to calculate.
                May take either value of `1` or `2`. Defaults to `1`.
            use_masses (bool): Utilize elemental masses in calculating the NPR. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (dict): Keyword arguments for conformer generation.
        """
        variant = variant if isinstance(variant, int) else variant.lower()

        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        if variant not in list(range(1, 3)) + ["all"]:
            raise ValueError("Argument `variant` must have a value of either `1`, `2`, or `all`.")

        self.variant = variant

        self.FUNCTION_MAP = {
            1: Descriptors3D.NPR1,
            2: Descriptors3D.NPR2,
        }

    def get_names(self) -> List[Dict[str, str]]:
        names = []
        for label in self._parse_labels():
            if "1" in label:
                names.append("first")
            elif "2" in label:
                names.append("second")
            elif "3" in label:
                names.append("third")
        name = " normalized principal moments ratio (NPR)"

        return [{"noun": join_list_elements(names) + name}]

    def _parse_labels(self) -> List[str]:
        """
        Parse featurizer labels.

        Args:
            None.

        Returns:
            None.
        """
        if self.variant == "all":
            return [f"npr{i}_value" for i in range(1, 3)]
        return [f"npr{self.variant}_value"]

    def feature_labels(self) -> List[str]:
        return self._parse_labels()

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract NPR value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing value for NPR.
        """
        mol = molecule.rdkit_mol

        mol = self._get_conformer(mol)

        npr_function = self.FUNCTION_MAP.get(self.variant, self._measure_all)
        npr_value = npr_function(mol, force=self.force, useAtomicMasses=self.use_masses)
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


class PMIFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return the normalized principal moments ratio (NPR) value of a molecule."""

    def __init__(
        self,
        variant: Union[int, str] = "all",
        use_masses: bool = True,
        force=True,
        conformer_generation_kwargs=None,
    ):
        """Initialize class object.

        Args:
           variant(int): Variant of principal moments of inertia (PMI) to calculate.
                May take either value of `1`, `2`, or `3`. Defaults to `1`.
            use_masses (bool): Utilize elemental masses in calculating the PMI. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (dict): Keyword arguments for conformer generation.
        """

        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        variant = variant if isinstance(variant, int) else variant.lower()

        if variant not in list(range(1, 4)) + ["all"]:
            raise ValueError("Argument `pmi` must have a value of either `1`, `2`, `3`, or `all`.")

        self.variant = variant

        self.FUNCTION_MAP = {1: Descriptors3D.PMI1, 2: Descriptors3D.PMI2, 3: Descriptors3D.PMI3}

    def _parse_labels(self) -> List[str]:
        """
        Parse featurizer labels.

        Args:
            None.

        Returns:
            None.
        """
        if self.variant == "all":
            return [f"pmi{i}_value" for i in range(1, 4)]
        return [f"pmi{self.variant}_value"]

    def feature_labels(self) -> List[str]:
        return self._parse_labels()

    def get_names(self) -> List[Dict[str, str]]:
        names = []
        for label in self._parse_labels():
            if "1" in label:
                names.append("first")
            elif "2" in label:
                names.append("second")
            elif "3" in label:
                names.append("third")
        name = (
            " principal moment of inertia (PMI)"
            if len(names) == 1
            else " principal moments of inertia (PMI)"
        )
        return [{"noun": join_list_elements(names) + name}]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract PMI value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing value for PMI.
        """
        mol = molecule.rdkit_mol

        mol = self._get_conformer(mol)

        pmi_function = self.FUNCTION_MAP.get(self.variant, self._measure_all)

        pmi_value = pmi_function(mol, force=self.force, useAtomicMasses=self.use_masses)
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


class SpherocityIndexFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return the spherocity index of a molecule."""

    def __init__(self, use_masses: bool = True, force=True, conformer_generation_kwargs=None):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (dict): Keyword arguments for conformer generation.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "spherocity index"}]

    def feature_labels(self) -> List[str]:
        return ["spherocity_index"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract spherocity index for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing spherocity index value.
        """
        mol = molecule.rdkit_mol

        mol = self._get_conformer(mol)

        spherocity_index = Descriptors3D.SpherocityIndex(
            mol, force=self.force, useAtomicMasses=self.use_masses
        )
        return np.array([spherocity_index]).reshape(1, -1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class RadiusOfGyrationFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return the radius of gyration of a molecule."""

    def __init__(self, use_masses: bool = True, force=True, conformer_generation_kwargs=None):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (dict): Keyword arguments for conformer generation.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "radius of gyration"}]

    def feature_labels(self) -> List[str]:
        return ["radius_of_gyration"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract radius of gyration for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing radius of gyration.
        """
        mol = molecule.rdkit_mol

        mol = self._get_conformer(mol)

        gyration_radius = Descriptors3D.RadiusOfGyration(
            mol, force=self.force, useAtomicMasses=self.use_masses
        )
        return np.array([gyration_radius]).reshape(1, -1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
