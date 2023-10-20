# -*- coding: utf-8 -*-

"""Featurizers for proton- and electron-related information."""

from typing import Any, Dict, List, Optional, Union

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
    "AtomChargeFeaturizer",
    "AtomNucleophilicityFeaturizer",
    "AtomElectrophilicityFeaturizer",
    "MoleculeNucleophilicityFeaturizer",
    "MoleculeElectrophilicityFeaturizer",
    "MoleculeNucleofugalityFeaturizer",
    "MoleculeElectrofugalityFeaturizer",
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
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
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
            None.

        Returns:
            (List[str]): List of implementors.
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
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
        return ["num_hydrogen_bond_donors"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the number of Hydrogen bond donors present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (np.array): Number of Hydrogen bond donors present in `molecule`.
        """
        return np.array([rdMolDescriptors.CalcNumHBD(molecule.reveal_hydrogens())]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer to obtain molecular valence electron count"""


class ValenceElectronCountFeaturizer(AbstractFeaturizer):
    """A featurizer for extracting valence electron count."""

    def __init__(self):
        """Initialize class.

        Args:
            None.
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
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
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
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ElectronAffinityFeaturizer(MorfeusFeaturizer):
    """Featurize molecule and return electron affinity."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
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
            (List[str]): List of labels of extracted features.
        """
        return ["electron_affinity"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class IonizationPotentialFeaturizer(MorfeusFeaturizer):
    """Featurize molecule and return ionization potential."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
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
            (List[str]): List of labels of extracted features.
        """
        return ["ionization_potential"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class HOMOEnergyFeaturizer(MorfeusFeaturizer):
    """Featurize molecule and return energy of highest occupied molecular orbital."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
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
        return np.array([xtb.get_homo()]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return ["homo_energy"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class LUMOEnergyFeaturizer(MorfeusFeaturizer):
    """Featurize molecule and return energy of lowest unoccupied molecular orbital."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
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
        return np.array([xtb.get_lumo()]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return ["lumo_energy"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class AtomChargeFeaturizer(MorfeusFeaturizer):
    """Return the charges for atoms in molecules."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
        atom_indices: Union[int, List[int]] = 100,
        as_range: bool = False,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            atom_indices (Union[int, List[int]]): Range of atoms to calculate areas for. Either:
                - an integer,
                - a list of integers, or
                - a two-tuple of integers representing lower index and upper index.
            as_range (bool): Use `atom_index_range` parameter as a range of indices or not. Defaults to `False`
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "atom charges",
            },
        ]

        self.atom_indices, self.as_range = self._parse_indices(atom_indices, as_range)

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing charges for atoms in molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")

        atom_charges = morfeus_instance.get_charges()
        num_atoms = len(atom_charges)

        atom_areas = [(atom_charges[i] if i <= num_atoms else 0) for i in self.atom_indices]

        return np.array(atom_areas).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return [f"atom_charge_{i}" for i in self.atom_indices]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class AtomNucleophilicityFeaturizer(MorfeusFeaturizer):
    """Return the nucleophilicity value for each atom in a molecule."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
        atom_indices: Union[int, List[int]] = 100,
        as_range: bool = False,
        local: bool = False,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            atom_indices (Union[int, List[int]]): Range of atoms to calculate areas for. Either:
                - an integer,
                - a list of integers, or
                - a two-tuple of integers representing lower index and upper index.
            as_range (bool): Use `atom_indices` parameter as a range of indices or not. Defaults to `False`.
            local (bool): Calculate local descriptor or not. Defaults to `False`.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "local fukui nucleophilicity" if local else "fukui nucleophilicity",
            },
        ]

        self.local = local

        self.atom_indices, self.as_range = self._parse_indices(atom_indices, as_range)

    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing nucleophilicity value for each atom in a molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")
        descriptor = "local_nucleophilicity" if self.local else "nucleophilicity"

        nucleophilicity = morfeus_instance.get_fukui(descriptor)
        num_atoms = len(nucleophilicity)

        atom_nucleophilicities = [
            (nucleophilicity[i] if i <= num_atoms else 0) for i in self.atom_indices
        ]

        return np.array(atom_nucleophilicities).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return [
            (f"atom_{i}_local_nucleophilicity" if self.local else f"atom_{i}_nucleophilicity")
            for i in self.atom_indices
        ]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class AtomElectrophilicityFeaturizer(MorfeusFeaturizer):
    """Return electrophilicity values for each atom in a molecule."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
        atom_indices: Union[int, List[int]] = 100,
        as_range: bool = False,
        local: bool = False,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            atom_indices (Union[int, List[int]]): Range of atoms to calculate areas for. Either:
                - an integer,
                - a list of integers, or
                - a two-tuple of integers representing lower index and upper index.
            as_range (bool): Use `atom_indices` parameter as a range of indices or not. Defaults to `False`.
            local (bool): Calculate local descriptor or not. Defaults to `False`.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "local fukui electrophilicity" if local else "fukui electrophilicity",
            },
        ]

        self.local = local

        self.atom_indices, self.as_range = self._parse_indices(atom_indices, as_range)

    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing electrophilicity value for each atom in a molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")
        descriptor = "local_electrophilicity" if self.local else "electrophilicity"

        electrophilicity = morfeus_instance.get_fukui(descriptor)
        num_atoms = len(electrophilicity)

        atom_electrophilicities = [
            (electrophilicity[i] if i <= num_atoms else 0) for i in self.atom_indices
        ]

        return np.array(atom_electrophilicities).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return [
            (f"atom_{i}_local_electrophilicity" if self.local else f"atom_{i}_electrophilicity")
            for i in self.atom_indices
        ]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MoleculeNucleophilicityFeaturizer(MorfeusFeaturizer):
    """Return the global nucleophilicity value for a molecule."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "global nucleophilicity",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing global nucleophilicity value for the molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")

        nucleophilicity = morfeus_instance.get_global_descriptor(
            "nucleophilicity", **self.morfeus_kwargs
        )

        return np.array([nucleophilicity]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return ["molecular_nucleophilicity"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MoleculeElectrophilicityFeaturizer(MorfeusFeaturizer):
    """Return global electrophilicity value for a molecule."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "global electrophilicity",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing global electrophilicity value for the molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")

        electrophilicity = morfeus_instance.get_global_descriptor(
            "electrophilicity", **self.morfeus_kwargs
        )

        return np.array([electrophilicity]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return ["molecular_electrophilicity"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MoleculeNucleofugalityFeaturizer(MorfeusFeaturizer):
    """Return the global nucleofugality value for a molecule."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "global nucleofugality",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing global nucleofugality value for the molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")

        nucleofugality = morfeus_instance.get_global_descriptor(
            "nucleofugality", **self.morfeus_kwargs
        )

        return np.array([nucleofugality]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return ["molecular_nucleofugality"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MoleculeElectrofugalityFeaturizer(MorfeusFeaturizer):
    """Return the global electrofugality value for a molecule."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "global electrofugality",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing global electrofugality value for the molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")

        electrofugality = morfeus_instance.get_global_descriptor(
            "electrofugality", **self.morfeus_kwargs
        )

        return np.array([electrofugality]).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return ["molecular_electrofugality"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
