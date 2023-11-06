# -*- coding: utf-8 -*-

"""Featurizers for chemical bond-related information."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from rdkit.Chem import rdMolDescriptors

from chemcaption.featurize.base import AbstractFeaturizer, MorfeusFeaturizer
from chemcaption.featurize.utils import join_list_elements
from chemcaption.molecules import Molecule

# Implemented bond-related featurizers

__all__ = [
    "RotableBondCountFeaturizer",
    "RotableBondProportionFeaturizer",
    "BondTypeCountFeaturizer",
    "BondTypeProportionFeaturizer",
    "DipoleMomentsFeaturizer",
    "BondOrderFeaturizer",
]


"""Featurizer for counting rotatable bonds in molecule."""

# compared to "all" implemented rdkit bond types we drop
# some of the dative bonds
_MAP_BOND_TYPE_TO_CLEAN_NAME = {
    "num_unspecified_bond": "unspecified",
    "num_single_bonds": "single",
    "num_double_bonds": "double",
    "num_triple_bonds": "triple",
    "num_quadruple_bonds": "quadruple",
    "num_quintuple_bonds": "quintuple",
    "num_hextuple_bonds": "hextuple",
    "num_oneandahalf_bonds": "one-and-a-half",
    "num_twoandahalf_bonds": "two-and-a-half",
    "num_threeandahalf_bonds": "three-and-a-half",
    "num_fourandahalf_bonds": "four-and-a-half",
    "num_fiveandahalf_bonds": "five-and-a-half",
    "num_aromatic_bonds": "aromatic",
    "num_ionic_bonds": "ionic",
    "num_hydrogen_bonds": "hydrogen",
    "num_threecenter_bonds": "three-center",
    "num_dativeone_bonds": "dative one-electron",
    "num_dative_bonds": "dative two-electron",
    "num_other_bonds": "other",
    "num_zero_bonds": "zero-order",
    "num_bonds": "total number of bonds",
}


class RotableBondCountFeaturizer(AbstractFeaturizer):
    """Obtain number of rotable (i.e., single, non-terminal, non-hydrogen) bonds in a molecule."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()

        self._names = [
            {
                "noun": "number of rotatable bonds",
            }
        ]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
        return ["num_rotable_bonds"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Count the number of rotable (single, non-terminal) bonds in a molecule.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            num_rotable (np.array): Number of rotable bonds in molecule.
        """
        num_rotable = rdMolDescriptors.CalcNumRotatableBonds(
            molecule.reveal_hydrogens(), strict=True
        )
        return np.array([num_rotable]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer for calculating distribution of molecule bonds between rotatable and non-rotatable bonds."""


class RotableBondProportionFeaturizer(AbstractFeaturizer):
    """Obtain distribution between rotable and non-rotable bonds in a molecule."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()

        self._names = [
            {
                "noun": "proportion of rotatable and non-rotatable bonds",
            }
        ]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return ["rotable_proportion", "non_rotable_proportion"]

    def _get_bond_types(self, molecule: Molecule) -> List[float]:
        """Return distribution of bonds based on rotability.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            bond_distribution (List[float]): Distribution of bonds based on rotability.
        """
        num_bonds = len(molecule.reveal_hydrogens().GetBonds())
        num_rotable = rdMolDescriptors.CalcNumRotatableBonds(
            molecule.reveal_hydrogens(), strict=False
        )
        num_non_rotable = num_bonds - num_rotable

        bond_distribution = [num_rotable / num_bonds, num_non_rotable / num_bonds]

        return bond_distribution

    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single molecule instance.
        Return distribution of bonds based on rotability.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (np.array): Array containing distribution of the bonds based on rotability.
        """
        return np.array(self._get_bond_types(molecule=molecule)).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer for calculating number of molecule bond types."""


class BondTypeCountFeaturizer(AbstractFeaturizer):
    """Featurizer for bond type count (or presence) extraction."""

    def __init__(self, count: bool = True, bond_type: Union[str, List[str]] = "all"):
        """
        Initialize class.

        Args:
            count (bool): If set to True, count pattern frequency.
                Otherwise, only encode presence. Defaults to True.
            bond_type (Union[str, List[str]]): Type of bond to enumerate.
                If `all`, enumerates all bonds irrespective of type. Default (ALL).
        """
        super().__init__()

        self.count = count
        self.prefix = "num_" if self.count else ""
        self.suffix = "_bonds" if self.count else "_bond_presence"
        self.prompt_template = (
            "Question: {PROPERTY_NAME} in the molecule with {REPR_SYSTEM} {REPR_STRING}?"
        )

        if self.count:
            self.constraint = "Constraint: Return a list of comma separated integers."
        else:
            self.constraint = "Constraint: Return a list of comma separated integer boolean indicators (0 for absence, 1 for presence)."
        self.bond_type = (
            [bond_type.upper()] if isinstance(bond_type, str) else [b.upper() for b in bond_type]
        )

    def _count_bonds(self, molecule: Molecule) -> List[int]:
        """
        Count the frequency of appearance for bond_type in a molecule.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            num_bonds (List[int]): Number of occurrences of `bond_type` in molecule.
        """
        all_bonds = self._get_bonds(molecule)

        bond_types, index = self._get_bond_types(), 1 if self.count else 0

        num_bonds = [
            all_bonds.count(bond_type.split("_")[index].upper())
            for bond_type in bond_types
            if bond_type != "num_bonds"
        ]

        if self.count and ("ALL" in self.bond_type):
            num_bonds.append(len(all_bonds))
        elif not self.count:
            num_bonds = [min(1, count) for count in num_bonds]

        return num_bonds

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return self._get_bond_types()

    def _get_bond_types(self) -> List[str]:
        """
        Return bond types.

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        if "ALL" in self.bond_type:
            bond_types = self._rdkit_bond_types()

            if self.count:
                bond_types.append("num_bonds")
        else:
            bond_types = self._parse_bond_names(self.bond_type)

        return bond_types

    def _parse_bond_names(self, bond_names: Union[str, List[str]]) -> List[str]:
        """Parse bond names for use in counting.

        Args:
            bond_names (Union[str, List[str]]): Bond names.

        Returns:
            (List[str]): Parsed bond names.
        """
        if isinstance(bond_names, str):
            bond_names = [self.prefix + bond_names.lower() + self.suffix]
        else:
            bond_names = [self.prefix + name.lower() + self.suffix for name in bond_names]
        return bond_names

    def get_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        mapped_names = [
            _MAP_BOND_TYPE_TO_CLEAN_NAME[bond_type]
            for bond_type in self._get_bond_types()
            if "num_bonds" != bond_type
        ]

        if self.count:
            name = "What is the number of "
        else:
            name = "Are there "

        return [{"noun": name + join_list_elements(mapped_names) + " bonds"}]

    def _get_bonds(
        self,
        molecule: Molecule,
    ) -> List[str]:
        """
        Extract all individual bonds present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            bonds (List[str]): List of all bonds present in molecule.
        """
        bonds = [
            str(bond.GetBondType()).split(".")[-1]
            for bond in molecule.reveal_hydrogens().GetBonds()
        ]

        return bonds

    def _rdkit_bond_types(self) -> List[str]:
        """
        Returns a list of bonds supported by rdkit.

        Args:
            None.

        Returns:
            (List[str]): List of all bonds present in rdkit.
        """
        return [k for k in _MAP_BOND_TYPE_TO_CLEAN_NAME.keys() if "num_bonds" != k]

    def _get_unique_bond_types(self, molecule: Molecule) -> List[str]:
        """
        Get the unique bond types present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            unique_bonds (List[str]): Set of unique bonds present in `molecule`.
        """
        bonds = self._get_bonds(molecule)
        unique_bonds = list(set(bonds))

        return unique_bonds

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Return integer array representing the:
            - frequency or
            - presence
            of bond types in a molecule.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing integer counts/signifier of bond type(s).
        """
        return np.array(self._count_bonds(molecule=molecule)).reshape(1, -1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer for calculating proportion of molecule bond types."""


class BondTypeProportionFeaturizer(BondTypeCountFeaturizer):
    """Featurizer for bond type proportion extraction."""

    def __init__(self, bond_type: Union[str, List[str]] = "all"):
        """
        Initialize class.

        Args:
            bond_type (Union[str, List[str]]): Type of bond to enumerate.
                If `all`, enumerates all bonds irrespective of type. Default (ALL).
        """
        super().__init__(count=True, bond_type=bond_type)
        self.constraint = "Constraint: Return a list of comma separated floats."
        self.prefix = ""
        self.suffix = "_bond_proportion"

    def get_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        mapped_names = [
            _MAP_BOND_TYPE_TO_CLEAN_NAME[bond_type] for bond_type in super().feature_labels()
        ]
        return [
            {"noun": "What is the proportion of " + join_list_elements(mapped_names) + " bonds"}
        ]

    def _get_bond_distribution(self, molecule: Molecule) -> List[float]:
        """Return a frequency distribution for the bonds present in a molecule.
        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            bond_distribution (List[float]): List of bond type proportions.
        """
        num_bonds = self._count_bonds(molecule=molecule)

        total_bond_count = (
            num_bonds.pop(-1)
            if "ALL" in self.bond_type
            else len(self._get_bonds(molecule=molecule))
        )

        bond_proportion = [count / total_bond_count for count in num_bonds]

        return bond_proportion

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Return float(s) containing on bond type proportion(s).

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing bond type proportion(s).
        """
        return np.array(self._get_bond_distribution(molecule=molecule)).reshape(1, -1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        labels = [label for label in super().feature_labels() if label != "num_bonds"]
        labels = self._parse_bond_names([x.split("_")[1] for x in labels])

        return labels

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class DipoleMomentsFeaturizer(MorfeusFeaturizer):
    """Return the dipole moments for a molecule."""

    def __init__(
            self,
            conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
            morfeus_kwargs: Optional[Dict[str, Any]] = None,
            max_index: Union[int, List[int]] = 2,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            max_index (Union[int, List[int]]): Maximum number of atoms/bonds to consider for feature generation.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "dipole moments",
            },
        ]

        self.max_index = max_index

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing dipole moments for bonds in molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")

        dipoles = morfeus_instance.get_dipole(**self.morfeus_kwargs).flatten().tolist()
        num_dipoles = len(dipoles)

        dipoles = [(dipoles[i - 1] if i <= num_dipoles else 0) for i in range(self.max_index)]

        return np.array(dipoles).reshape(1, -1)

    def featurize_many(self, molecules: List[Molecule]) -> np.array:
        """
        Featurize a sequence of Molecule objects.

        Args:
            molecules (List[Molecule]): A sequence of molecule representations.

        Returns:
            (np.array): An array of features for each molecule instance.
        """
        self.max_index = self.fit_on_bond_counts(molecules=molecules)

        return super().featurize_many(molecules=molecules)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return [f"dipole_{i}_{i+1}" for i in range(self.max_index)]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class BondOrderFeaturizer(MorfeusFeaturizer):
    """Return the bond orders for bonds in a molecule."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
        max_index: Union[int, List[int]] = 2,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            max_index (Union[int, List[int]]): Maximum number of atoms/bonds to consider for feature generation.
        """
        super().__init__(
            conformer_generation_kwargs=conformer_generation_kwargs,
            morfeus_kwargs=morfeus_kwargs,
        )

        self._names = [
            {
                "noun": "bond orders",
            },
        ]

        self.max_index = max_index

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing bond orders for bonds in molecule instance.
        """
        morfeus_instance = self._get_morfeus_instance(molecule=molecule, morpheus_instance="xtb")

        bond_orders = morfeus_instance.get_bond_orders(**self.morfeus_kwargs).flatten().tolist()

        bond_orders = [(bond_orders[i - 1] if i <= self.max_index else 0) for i in range(self.max_index)]

        return np.array(bond_orders).reshape(1, -1)

    def featurize_many(self, molecules: List[Molecule]) -> np.array:
        """
        Featurize a sequence of Molecule objects.

        Args:
            molecules (List[Molecule]): A sequence of molecule representations.

        Returns:
            (np.array): An array of features for each molecule instance.
        """
        self.max_index = self.fit_on_bond_counts(molecules=molecules)

        return super().featurize_many(molecules=molecules)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels of extracted features.
        """
        return [f"bond_order_{i}_{i+1}" for i in range(self.max_index)]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
