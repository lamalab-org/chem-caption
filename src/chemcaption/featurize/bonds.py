# -*- coding: utf-8 -*-

"""Featurizers for chemical bond-related information."""

from typing import List, Union

import numpy as np
from rdkit.Chem import rdMolDescriptors

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import Molecule
from chemcaption.featurize.utils import join_list_elements

# Implemented bond-related featurizers

__all__ = [
    "RotableBondCountFeaturizer",
    "BondRotabilityFeaturizer",
    "BondTypeCountFeaturizer",
    "BondTypeProportionFeaturizer",
]


"""Featurizer for counting rotatable bonds in molecule."""

# compared to "all" implemented rdkit bond types we drop
# some of the dative bonds
_MAP_BOND_TYPE_TO_CLEAN_NAME = {
    "num_unspecified_bond": "number of unspecified bonds",
    "num_single_bonds": "number of single bonds",
    "num_double_bonds": "number of double bonds",
    "num_triple_bonds": "number of triple bonds",
    "num_quadruple_bonds": "number of quadruple bonds",
    "num_quintuple_bonds": "number of quintuble bonds",
    "num_hextuple_bonds": "number of hextuple bonds",
    "num_oneandahalf_bonds": "number of one-and-a-half bonds",
    "num_twoandahalf_bonds": "number of two-and-a-half bonds",
    "num_threeandahalf_bonds": "number of three-and-a-half bonds",
    "num_fourandahalf_bonds": "number of four-and-a-half bonds",
    "num_fiveandahalf_bonds": "number of five-and-a-half bonds",
    "num_aromatic_bonds": "number of aromatic bonds",
    "num_ionic_bonds": "number of ionic bonds",
    "num_hydrogen_bonds": "number of hydrogen bonds",
    "num_threecenter_bonds": "number of three-center bonds",
    "num_dativeone_bonds": "number of dative one-electron bonds",
    "num_dative_bonds": "number of two-electron dative bonds",
    "num_other_bonds": "number of other bonds",
    "num_zero_bonds": "number of zero-order bonds",
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
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer for calculating distribution of molecule bonds between rotatable and non-rotatable bonds."""


class BondRotabilityFeaturizer(AbstractFeaturizer):
    """Obtain distribution between rotable and non-rotable bonds in a molecule."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()

        self.template = (
            "What are the {PROPERTY_NAME} for the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "proportions of rotatable and non-rotatable bonds",
            }
        ]

    def feature_labels(self) -> List[str]:
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
        """Featurize single molecule instance. Return distribution of bonds based on rotability.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: Array containing distribution of the bonds based on rotability.
        """
        return np.array(self._get_bond_types(molecule=molecule)).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer for calculating number of molecule bond types."""


class BondTypeCountFeaturizer(AbstractFeaturizer):
    """Featurizer for bond type count (or presence) extraction."""

    def __init__(self, count: bool = True, bond_type: Union[str, List[str]] = "all"):
        """
        Initialize class.

        Args:
            count (bool): If set to True, count pattern frequency. Otherwise, only encode presence. Defaults to True.
            bond_type (Union[str, List[str]]): Type of bond to enumerate.
                If `all`, enumerates all bonds irrespective of type. Default (ALL).
        """
        super().__init__()

        self.count = count
        self.prefix = "num_" if self.count else ""
        self.suffix = "_bonds" if self.count else "_bond_presence"

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

        num_bonds = []

        bond_types = self.feature_labels()

        num_bonds = [
            all_bonds.count(bond_type.split("_")[1].upper())
            for bond_type in bond_types
            if bond_type != "num_bonds"
        ]

        if self.count:
            num_bonds.append(len(all_bonds))

        if not self.count:
            num_bonds = [min(1, count) for count in num_bonds]

        return num_bonds

    def feature_labels(self) -> List[str]:
        """
        Parse featurizer labels.

        Args:
            None.

        Returns:
            bond_types (List[str]): List of strings of bond types.
        """
        label = []

        if "ALL" in self.bond_type:
            bond_types = self._rdkit_bond_types()

            if self.count:
                label.append("num_bonds")
        else:
            bond_types = self.bond_type

        return bond_types

    def get_names(self) -> List[str]:
        mapped_names = [
            _MAP_BOND_TYPE_TO_CLEAN_NAME[bond_type] for bond_type in self.feature_labels()
        ]
        return [{"noun": join_list_elements(mapped_names)}]

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
        return list(_MAP_BOND_TYPE_TO_CLEAN_NAME.keys())

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
            List[str]: List of implementors.
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

        self.prefix = ""
        self.suffix = "_bond_proportion"

        _ = self._parse_labels()

    def _get_bond_distribution(self, molecule: Molecule) -> List[float]:
        """Return a frequency distribution for the bonds present in a molecule.
        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            bond_distribution (List[float]): List of bond type proportions.
        """
        num_bonds = self._count_bonds(molecule=molecule)

        total_bond_count = (
            sum(num_bonds) if "ALL" in self.bond_type else len(self._get_bonds(molecule=molecule))
        )

        bond_proportion = [count / total_bond_count for count in num_bonds]

        return bond_proportion

    def _count_bonds(self, molecule: Molecule) -> List[int]:
        """
        Count the frequency of appearance for bond_type in a molecule.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            num_bonds (List[int]): Number of occurrences of `bond_type` in molecule.
        """
        num_bonds = super()._count_bonds(molecule=molecule)
        num_bonds = num_bonds[:-1] if "ALL" in self.bond_type else num_bonds

        return num_bonds

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Return float(s) containing on bond type proportion(s).

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing bond type proportion(s).
        """
        return np.array(self._get_bond_distribution(molecule=molecule)).reshape(1, -1)
