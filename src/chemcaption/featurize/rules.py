# -*- coding: utf-8 -*-

"""Featurizers for drug & molecular rules."""

from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

# Implemented drug rule-related featurizers.

__all__ = [
    "LipinskiViolationsFeaturizer",
]


class LipinskiViolationsFeaturizer(AbstractFeaturizer):
    """Returns the number of violations of Lipinski's Rule of 5."""

    def __init__(self):
        """Instantiate class."""
        super().__init__()
        self.label = ["num_lipinski_violations"]

    def _mass_violation(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """Return molecule status as regards violation of Lipinski's molar mass rule (must be < 500 Daltons).

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        molar_mass = Descriptors.ExactMolWt(molecule.rdkit_mol)
        return np.array([molar_mass > 500], dtype=int).reshape((1, -1))

    def _hydrogen_bond_donor_violation(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """Return molecule status as regards violation of Lipinski's hydrogen bond donor rule (must be < 5).

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        hbd = Chem.Lipinski.NumHDonors(molecule.rdkit_mol)
        return np.array([hbd > 5], dtype=int).reshape((1, -1))

    def _hydrogen_bond_acceptor_violation(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """Return molecule status as regards violation of Lipinski's hydrogen bond acceptor rule (must be < 10).

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        hba = Chem.Lipinski.NumHAcceptors(molecule.rdkit_mol)
        return np.array([hba > 10], dtype=int).reshape((1, -1))

    def _log_p_violation(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """Return molecule status as regards violation of Lipinski's LogP rule (must be < 5).

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        log_p = Descriptors.MolLogP(molecule.rdkit_mol)
        return np.array([log_p > 5], dtype=int).reshape((1, -1))

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance. Returns the number of Lipinski rules violated by a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            (np.array): number of Lipinski Rule of 5 violations.
        """
        num_violations = (
            self._mass_violation(molecule)
            + self._log_p_violation(molecule)
            + self._hydrogen_bond_acceptor_violation(molecule)
            + self._hydrogen_bond_acceptor_violation(molecule)
        )

        return np.array([num_violations]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]