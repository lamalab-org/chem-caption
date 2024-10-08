# -*- coding: utf-8 -*-

"""Featurizers for 3D (i.e., spatial) features."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from frozendict import frozendict
from rdkit import Chem
from rdkit.Chem import Descriptors3D

from chemcaption.featurize.base import AbstractFeaturizer, MorfeusFeaturizer
from chemcaption.featurize.utils import cached_conformer, join_list_elements
from chemcaption.molecules import Molecule

# Implemented spatial featurizers

__all__ = [
    "SpatialFeaturizer",
    "EccentricityFeaturizer",
    "AsphericityFeaturizer",
    "InertialShapeFactorFeaturizer",
    "NPRFeaturizer",
    "PMIFeaturizer",
    "AtomVolumeFeaturizer",
    "SpherocityIndexFeaturizer",
    "RadiusOfGyrationFeaturizer",
]


"""Abstract Featurizer for extracting 3D features from molecule."""


class SpatialFeaturizer(AbstractFeaturizer):
    """Abstract class for 3-D featurizers."""

    def __init__(
        self,
        use_masses: bool = True,
        force: bool = True,
        conformer_generation_kwargs: Optional[Dict[str, Union[int, str]]] = None,
    ):
        """Instantiate initialization scheme to be inherited.

        Args:
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization. Defaults to `True`.
            conformer_generation_kwargs (Optional[Dict[str, Union[int, str]]]):
                Keyword arguments for conformer generation. Defaults to `None`.
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

    def _get_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        """Returns molecular object embedded with conformers.

        Args:
            mol (Chem.Mol): Rdkit molecular instance.

        Returns:
            (Chem.Mol): Rdkit molecular instance embedded with conformers.
        """
        smiles = Chem.MolToSmiles(mol)
        return cached_conformer(smiles, self._conf_gen_kwargs)

    def _base_rdkit_utility_keys(self) -> List[str]:
        """Returns sorted identifiers for `rdkit` functions in function map.

        Args:
            None.

        Returns:
            List[str]: List of ordered function keys.
        """
        keys = list(k for k in self.FUNCTION_MAP.keys())
        keys.sort()
        return keys

    def _measure_all(
        self, *x: Chem.Mol, **y: Dict[str, Union[int, str]]
    ) -> List[Union[int, float]]:
        """Return results for all possible variants of 3D featurizer.

        Args:
            *x (Chem.Mol): rdkit Molecule object.
            **y (Dict[str, Union[int, str]]): Keyword arguments.

        Returns:
            List[Union[int, float]]: List of computed results for different variants of interest.
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
            NotImplementedError: Exception signifying lack of implementation.
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


class EccentricityFeaturizer(SpatialFeaturizer):
    """Featurizer to return eccentricity value of a molecule."""

    def __init__(
        self,
        use_masses: bool = True,
        force: bool = True,
        conformer_generation_kwargs: Optional[Dict[str, Union[int, str]]] = None,
    ):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization. Defaults to `True`.
            conformer_generation_kwargs (Optional[Dict[str, Union[int, str]]]):
                Keyword arguments for conformer generation. Defaults to `None`.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "eccentricity"}]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels of extracted features.
        """
        return ["eccentricity"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract eccentricity value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.array: Array containing eccentricity value.
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
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class AsphericityFeaturizer(SpatialFeaturizer):
    """Featurizer to return number of asphericity value of a molecule."""

    def __init__(
        self,
        use_masses: bool = True,
        force: bool = True,
        conformer_generation_kwargs: Optional[Dict[str, Union[int, str]]] = None,
    ):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses in asphericity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization. Defaults to `True`.
            conformer_generation_kwargs (Optional[Dict[str, Union[int, str]]]):
                Keyword arguments for conformer generation. Defaults to `None`.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "asphericity"}]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels of extracted features.
        """
        return ["asphericity"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract asphericity value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.array: Array containing asphericity value.
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
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class InertialShapeFactorFeaturizer(SpatialFeaturizer):
    """Featurizer to return inertial shape factor of a molecule."""

    def __init__(
        self,
        use_masses: bool = True,
        force: bool = True,
        conformer_generation_kwargs: Optional[Dict[str, Union[int, str]]] = None,
    ):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses for calculating the inertial shape factor. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization. Defaults to `True`.
            conformer_generation_kwargs (Optional[Dict[str, Union[int, str]]]):
                Keyword arguments for conformer generation. Defaults to `None`.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "inertial shape factor"}]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels of extracted features.
        """
        return ["inertial_shape_factor"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract inertia shape factor for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.array: Array containing inertia shape factor.
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
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class NPRFeaturizer(SpatialFeaturizer):
    """Featurizer to return the Normalized principal moments ratio (NPR) value of a molecule."""

    def __init__(
        self,
        variant: Union[int, str] = "all",  # today make iterable
        use_masses: bool = True,
        force: bool = True,
        conformer_generation_kwargs: Optional[Dict[str, Union[int, str]]] = None,
    ):
        """Initialize class object.

        Args:
            variant (Union[int, str]): Variant of normalized principal moments ratio (NPR) to calculate.
                May take either value of `1`, `2`, or `all`. Defaults to `all`.
            use_masses (bool): Utilize elemental masses in calculating the NPR. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization. Defaults to `True`.
            conformer_generation_kwargs (Optional[Dict[str, Union[int, str]]]):
                Keyword arguments for conformer generation. Defaults to `None`.
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
        """Return feature names.

        Args:
            None.

        Returns:
            List[Dict[str, str]]: List of names for extracted features according to parts-of-speech.
        """
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
            List[str]: Generate labels for featurizer.
        """
        if self.variant == "all":
            return [f"npr{i}_value" for i in range(1, 3)]
        return [f"npr{self.variant}_value"]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels of extracted features.
        """
        return self._parse_labels()

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract NPR value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.array: Array containing value(s) for NPR.
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
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class PMIFeaturizer(SpatialFeaturizer):
    """Featurizer to return the normalized principal moments ratio (NPR) value of a molecule."""

    def __init__(
        self,
        variant: Union[int, str] = "all",
        use_masses: bool = True,
        force: bool = True,
        conformer_generation_kwargs: Optional[Dict[str, Union[int, str]]] = None,
    ):
        """Initialize class object.

        Args:
            variant (Union[int, str]): Variant of principal moments of inertia (PMI) to calculate.
                May take either value of `1`, `2`, `3`, or `all`. Defaults to `all`.
            use_masses (bool): Utilize elemental masses in calculating the PMI. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization. Defaults to `True`.
            conformer_generation_kwargs (Optional[Dict[str, Union[int, str]]]):
                Keyword arguments for conformer generation. Defaults to `None`.
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
            List[str]: Generate labels for featurizer.
        """
        if self.variant == "all":
            return [f"pmi{i}_value" for i in range(1, 4)]
        return [f"pmi{self.variant}_value"]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of names of extracted features.
        """
        return self._parse_labels()

    def get_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            List[Dict[str, str]]: List of names for extracted features according to parts-of-speech.
        """
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
            np.array: Array containing value(s) for PMI.
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


class AtomVolumeFeaturizer(MorfeusFeaturizer):
    """Return the solvent accessible volume per atom in molecule."""

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
                Redundant if `aggregation` is not `None`.
            aggregation (Optional[Union[str, List[str]]]): Aggregation to use on generated descriptors.
                Defaults to `None`. If `None`, track atom/bond/molecular descriptors and identities.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
            qc_optimize=qc_optimize,
            aggregation=aggregation,
        )

        self._names = [
            {
                "noun": "solvent accessible atom volume",
            },
        ]

        self.max_index = max_index

    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing solvent accessible volumes for atoms in molecule instance.
        """
        if self.qc_optimize:
            molecule = self._generate_conformer(molecule=molecule)

        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="sasa")

        atom_volumes = morfeus_instance.atom_volumes
        num_atoms = len(atom_volumes)

        if self.max_index is None:
            self.max_index = self.fit_on_atom_counts(molecules=molecule)

        atom_volumes = [
            (atom_volumes[i] if i <= num_atoms else 0) for i in range(1, self.max_index + 1)
        ]

        if self.aggregation is None:
            # Track atom identities
            atomic_numbers = self._track_atom_identity(molecule=molecule, max_index=self.max_index)

            # Combine descriptors with atom identities
            atom_volumes = atom_volumes + atomic_numbers
        else:
            if isinstance(self.aggregation, (list, tuple, set)):
                atom_volumes = [
                    self.aggregation_func[agg](atom_volumes) for agg in self.aggregation
                ]
            else:
                atom_volumes = self.aggregation_func[self.aggregation](atom_volumes)

        return np.array(atom_volumes).reshape(1, -1)

    def featurize_many(self, molecules: List[Molecule]) -> np.array:
        """
        Featurize a sequence of Molecule objects.

        Args:
            molecules (List[Molecule]): A sequence of molecule representations.

        Returns:
            (np.array): An array of features for each molecule instance.
        """
        if self.max_index is None:
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
            return [f"solvent_accessible_atom_volume_{i}" for i in range(self.max_index)] + [
                f"atomic_number_{i}" for i in range(self.max_index)
            ]
        else:
            if isinstance(self.aggregation, (list, set, tuple)):
                return [f"solvent_accessible_atom_volume_{agg}" for agg in self.aggregation]
            else:
                return ["solvent_accessible_atom_volume_" + self.aggregation]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class SpherocityIndexFeaturizer(SpatialFeaturizer):
    """Featurizer to return the spherocity index of a molecule."""

    def __init__(
        self,
        use_masses: bool = True,
        force: bool = True,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Keyword arguments for conformer generation.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "spherocity index"}]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels of extracted features.
        """
        return ["spherocity_index"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract spherocity index for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.array: Array containing spherocity index value.
        """
        mol = molecule.rdkit_mol

        mol = self._get_conformer(mol)

        spherocity_index = Descriptors3D.SpherocityIndex(
            mol,
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


class RadiusOfGyrationFeaturizer(SpatialFeaturizer):
    """Featurizer to return the radius of gyration of a molecule."""

    def __init__(
        self,
        use_masses: bool = True,
        force: bool = True,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize class object.

        Args:
            use_masses (bool): Utilize elemental masses in eccentricity calculation. Defaults to `True`.
            force (bool): Utilize force field calculations for energy minimization.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Keyword arguments for conformer generation.
        """
        super().__init__(
            use_masses=use_masses,
            force=force,
            conformer_generation_kwargs=conformer_generation_kwargs,
        )

        self._names = [{"noun": "radius of gyration"}]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels of extracted features.
        """
        return ["radius_of_gyration"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract radius of gyration for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.array: Array containing the value for the radius of gyration.
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
