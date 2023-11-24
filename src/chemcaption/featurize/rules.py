# -*- coding: utf-8 -*-

"""Featurizers for drug & molecular rules."""

from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import Molecule

# Implemented drug rule-related featurizers.

__all__ = [
    "LipinskiFilterFeaturizer",
    "GhoseFilterFeaturizer",
    "LeadLikenessFilterFeaturizer",
]


class LipinskiFilterFeaturizer(AbstractFeaturizer):
    """Returns the number of violations of Lipinski's Rule of 5."""

    def __init__(self):
        """Instantiate class."""
        super().__init__()

        self._names = [
            {
                "noun": "number of Lipinski violations",
            }
        ]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["num_lipinski_violations"]

    def _mass_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of Lipinski's molar mass rule (must be < 500 Daltons).

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        molar_mass = Descriptors.ExactMolWt(molecule.rdkit_mol)
        return np.array([molar_mass > 500], dtype=int).reshape((1, -1))

    def _hydrogen_bond_donor_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of Lipinski's hydrogen bond donor rule (must be < 5).

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        hbd = Chem.Lipinski.NumHDonors(molecule.rdkit_mol)
        return np.array([hbd > 5], dtype=int).reshape((1, -1))

    def _hydrogen_bond_acceptor_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of Lipinski's hydrogen bond acceptor rule (must be < 10).

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        hba = Chem.Lipinski.NumHAcceptors(molecule.rdkit_mol)
        return np.array([hba > 10], dtype=int).reshape((1, -1))

    def _log_p_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of Lipinski's LogP rule (must be < 5).

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        log_p = Descriptors.MolLogP(molecule.rdkit_mol)
        return np.array([log_p > 5], dtype=int).reshape((1, -1))

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Returns the number of Lipinski rules violated by a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (np.array): number of Lipinski Rule of 5 violations.
        """
        num_violations = (
            self._mass_violation(molecule)
            + self._log_p_violation(molecule)
            + self._hydrogen_bond_acceptor_violation(molecule)
            + self._hydrogen_bond_acceptor_violation(molecule)
        ).astype(int)

        return num_violations

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class GhoseFilterFeaturizer(AbstractFeaturizer):
    """Returns the number of violations of Ghose filter."""

    def __init__(
        self,
        lower_mass: int = 160,
        upper_mass: int = 480,
        lower_logp: float = -0.4,
        upper_logp: float = 5.6,
        lower_atom_count: int = 20,
        upper_atom_count: int = 70,
        lower_refractivity: float = 40,
        upper_refractivity: float = 130,
    ):
        """Instantiate class.

        Args:
            lower_mass (int): Lower molar mass limit. Defaults to `160`.
            upper_mass (int): Upper molar mass limit. Defaults to `480`.
            lower_logp (float): Lower LogP limit. Defaults to `-0.4`.
            upper_logp (float): Upper LogP limit. Defaults to `5.6`.
            lower_atom_count (int): Lower limit for numer of atoms in molecule. Defaults to `20`.
            upper_atom_count (int): Upper limit for numer of atoms in molecule. Defaults to `70`.
            lower_refractivity (int): Lower limit for molecular refractivity. Defaults to `40`.
            upper_refractivity (int): Upper limit for molecular refractivity. Defaults to `130`.

        """
        super().__init__()

        self.lower_mass, self.upper_mass = lower_mass, upper_mass
        self.lower_logp, self.upper_logp = lower_logp, upper_logp
        self.lower_atom_count, self.upper_atom_count = lower_atom_count, upper_atom_count
        self.lower_refractivity, self.upper_refractivity = lower_refractivity, upper_refractivity

        self._names = [
            {
                "noun": "number of Ghose filter violations",
            }
        ]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["num_ghose_violations"]

    def _mass_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of Ghose filter molar mass rule.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        molar_mass = Descriptors.ExactMolWt(molecule.rdkit_mol)
        return np.array(
            [(molar_mass <= self.upper_mass) & (molar_mass >= self.lower_mass)], dtype=int
        ).reshape((1, -1))

    def _log_p_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of Ghose filter LogP rule.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        log_p = Chem.Crippen.MolLogP(molecule.rdkit_mol)
        return np.array(
            [(log_p >= self.lower_logp) and (log_p <= self.upper_logp)], dtype=float
        ).reshape((1, -1))

    def _atom_count_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of Ghose filter atom count rule.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        atom_count = len(molecule.reveal_hydrogens().GetAtoms())
        return np.array(
            [(atom_count >= self.lower_atom_count) and (atom_count <= self.upper_atom_count)],
            dtype=int,
        ).reshape((1, -1))

    def _refractivity_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of Ghose filter molar refractivity rule.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        refractivity = Chem.Crippen.MolMR(molecule.rdkit_mol)
        return np.array(
            [
                (refractivity >= self.lower_refractivity)
                and (refractivity <= self.upper_refractivity)
            ],
            dtype=int,
        ).reshape((1, -1))

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Returns the number of Ghose filter rules violated by a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (np.array): number of Ghose filter rule violations.
        """
        num_violations = (
            self._mass_violation(molecule)
            + self._log_p_violation(molecule)
            + self._atom_count_violation(molecule)
            + self._refractivity_violation(molecule)
        ).astype(int)

        return num_violations

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class LeadLikenessFilterFeaturizer(AbstractFeaturizer):
    """Returns the number of violations of lead-likeness filter."""

    def __init__(
        self,
        lower_mass: int = 250,
        upper_mass: int = 350,
        upper_logp: float = 3.5,
        upper_num_rotable_bonds: int = 7,
        strict_rotability: bool = True,
    ):
        """Instantiate class.

        Args:
            lower_mass (int): Lower molar mass limit. Defaults to `250`.
            upper_mass (int): Upper molar mass limit. Defaults to `350`.
            upper_logp (float): Upper LogP limit. Defaults to `3.5`.
            upper_num_rotable_bonds (int): Upper limit for number of rotatable bonds in molecule. Defaults to `7`.
            strict_rotability (bool): Calculate number of rotatable bonds by strict criterion. Defaults to `True`
        """
        super().__init__()

        self.lower_mass, self.upper_mass = lower_mass, upper_mass
        self.upper_logp = upper_logp
        self.lower_atom_count = upper_num_rotable_bonds
        self.upper_num_rotable_bonds = upper_num_rotable_bonds
        self.strict_rotability = strict_rotability

        self._names = [
            {
                "noun": "number of lead-likeness filter violations",
            }
        ]

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["num_lead_likeness_violations"]

    def _mass_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of molecular mass requirement for lead-likeness filter.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        molar_mass = Descriptors.ExactMolWt(molecule.rdkit_mol)
        return np.array(
            [(molar_mass <= self.upper_mass) & (molar_mass >= self.lower_mass)], dtype=int
        ).reshape((1, -1))

    def _log_p_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of LogP requirement for lead-likeness filter.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        log_p = Chem.Crippen.MolLogP(molecule.rdkit_mol)
        return np.array([(log_p <= self.upper_logp)], dtype=int).reshape((1, -1))

    def _rotable_bond_violation(self, molecule: Molecule) -> np.array:
        """Return molecule status as regards violation of rotatable bond count requirement for lead-likeness filter.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (np.array): integer representing violation status. 1 if rule is violated else 0.
        """
        num_rotable_bonds = rdMolDescriptors.CalcNumRotatableBonds(
            molecule.reveal_hydrogens(), strict=self.strict_rotability
        )
        return np.array([(num_rotable_bonds <= self.upper_num_rotable_bonds)], dtype=int).reshape(
            (1, -1)
        )

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Returns the number of lead-likeness filters violated by a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (np.array): number of lead-likeness filter violations.
        """
        num_violations = (
            self._mass_violation(molecule)
            + self._log_p_violation(molecule)
            + self._rotable_bond_violation(molecule)
        ).astype(int)

        return num_violations

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
