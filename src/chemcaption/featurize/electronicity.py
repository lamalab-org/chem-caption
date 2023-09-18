# -*- coding: utf-8 -*-

"""Featurizers for proton- and electron-related information."""

from typing import Any, Dict, List, Optional

import numpy as np
from rdkit.Chem import Descriptors, rdMolDescriptors

from chemcaption.featurize.base import AbstractFeaturizer, MorfeusFeaturizer
from chemcaption.molecules import Molecule

# Implemented proton-, electron- and charge-related featurizers

__all__ = [
    "HydrogenAcceptorCountFeaturizer",
    "HydrogenDonorCountFeaturizer",
    "ValenceElectronCountFeaturizer",
    "ElectronAffinityFeaturizer",
    "HOMOEnergyFeaturizer",
    "LUMOEnergyFeaturizer",
]


"""Featurizer to extract hydrogen acceptor count from molecules."""


class HydrogenAcceptorCountFeaturizer(AbstractFeaturizer):
    """Obtain number of Hydrogen bond acceptors in a molecule."""

    def __init__(self):
        """Get the number of Hydrogen bond acceptors present in a molecule."""
        super().__init__()

        self._names = [
            {
                "noun": "number of hydrogen bond acceptors",
            }
        ]

    def feature_labels(self) -> List[str]:
        return ["num_hydrogen_bond_acceptors"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the number of Hydrogen bond acceptors present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (np.array): Number of Hydrogen bond acceptors present in `molecule`.
        """
        return np.array([rdMolDescriptors.CalcNumHBA(molecule.reveal_hydrogens())]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer to extract hydrogen donor count from molecules."""


class HydrogenDonorCountFeaturizer(AbstractFeaturizer):
    """Obtain number of Hydrogen bond donors in a molecule."""

    def __init__(self):
        """Get the number of Hydrogen bond donors present in a molecule."""
        super().__init__()

        self._names = [
            {
                "noun": "number of hydrogen bond donors",
            }
        ]

    def feature_labels(self) -> List[str]:
        return ["num_hydrogen_bond_donors"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the number of Hydrogen bond donors present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: Number of Hydrogen bond donors present in `molecule`.
        """
        return np.array([rdMolDescriptors.CalcNumHBD(molecule.reveal_hydrogens())]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer to obtain molecular valence electron count"""


class ValenceElectronCountFeaturizer(AbstractFeaturizer):
    """A featurizer for extracting valence electron count."""

    def __init__(self):
        """Initialize class.

        Args:
            None
        """
        super().__init__()

        self._names = [
            {
                "noun": "number of valence electrons",
            },
            {
                "noun": "valence electron count",
            },
        ]

    def feature_labels(self) -> List[str]:
        return ["num_valence_electrons"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract and return valence electron count for molecular object.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing number of valence electrons.
        """
        num_valence_electrons = Descriptors.NumValenceElectrons(molecule.reveal_hydrogens())

        return np.array([num_valence_electrons]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ElectronAffinityFeaturizer(MorfeusFeaturizer):
    """Featurize molecule and return electron affinity."""

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
                "noun": "electron affinity",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing electron affinity for molecule instance.
        """
        xtb = self._get_morfeus_instance(molecule=molecule)
        return np.array([xtb.get_ea(**self.morfeus_kwargs)]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["electron_affinity"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class IonizationPotentialFeaturizer(MorfeusFeaturizer):
    """Featurize molecule and return ionization potential."""

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
                "noun": "ionization potential",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing ionization potential for molecule instance.
        """
        xtb = self._get_morfeus_instance(molecule=molecule)
        return np.array([xtb.get_ip(**self.morfeus_kwargs)]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["ionization_potential"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class HOMOEnergyFeaturizer(MorfeusFeaturizer):
    """Featurize molecule and return energy of highest occupied molecular orbital."""

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
                "noun": "energy of highest occupied molecular orbital",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing energy of highest occupied molecular orbital for molecule instance.
        """
        xtb = self._get_morfeus_instance(molecule=molecule)
        return np.array([xtb.get_homo(**self.morfeus_kwargs)]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["homo_energy"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class LUMOEnergyFeaturizer(MorfeusFeaturizer):
    """Featurize molecule and return energy of lowest unoccupied molecular orbital."""

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
                "noun": "energy of lowest unoccupied molecular orbital",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing energy of lowest unoccupied molecular orbital for molecule instance.
        """
        xtb = self._get_morfeus_instance(molecule=molecule)
        return np.array([xtb.get_lumo(**self.morfeus_kwargs)]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["lumo_energy"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
