# -*- coding: utf-8 -*-

"""Featurizers for chemical bond-related information."""

from typing import List, Union

import numpy as np
from rdkit.Chem import rdMolDescriptors

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

# Implemented bond-related featurizers

__all__ = [
    "NumRotableBondsFeaturizer",
    "BondRotabilityFeaturizer",
]


"""Featurizer for counting rotatable bonds in molecule."""


class NumRotableBondsFeaturizer(AbstractFeaturizer):
    """Obtain number of rotable (i.e., non-terminal, non-hydrogen) bonds in a molecule."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()
        self.label = ["num_rotable_bonds"]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Count the number of rotable (single, non-terminal) bonds in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

        Returns:
            num_rotable (np.array): Number of rotable bonds in molecule.
        """
        num_rotable = rdMolDescriptors.CalcNumRotatableBonds(molecule.rdkit_mol, strict=True)
        return np.array([num_rotable]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

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
        self.label = ["rotable_proportion", "non_rotable_proportion"]

    def _get_bond_types(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> List[float]:
        """Return distribution of bonds based on rotability.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            bond_distribution (List[float]): Distribution of bonds based on rotability.
        """
        num_bonds = len(molecule.rdkit_mol.GetBonds())
        num_rotable = rdMolDescriptors.CalcNumRotatableBonds(molecule.rdkit_mol, strict=False)
        num_non_rotable = num_bonds - num_rotable

        bond_distribution = [num_rotable / num_bonds, num_non_rotable / num_bonds]

        return bond_distribution

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """Featurize single molecule instance. Return distribution of bonds based on rotability.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: Array containing distribution of the bonds based on rotability.
        """
        return np.array(self._get_bond_types(molecule=molecule)).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
