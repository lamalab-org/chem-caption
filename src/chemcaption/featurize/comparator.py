# -*- coding: utf-8 -*-

"""Implementations for molecule Comparator utilities."""

from typing import List

import numpy as np

from chemcaption.featurize.base import AbstractFeaturizer, Comparator, MultipleComparator
from chemcaption.featurize.bonds import BondTypeCountFeaturizer
from chemcaption.featurize.composition import AtomCountFeaturizer, MolecularFormulaFeaturizer
from chemcaption.featurize.electronicity import ValenceElectronCountFeaturizer
from chemcaption.featurize.rules import (
    GhoseFilterFeaturizer,
    LeadLikenessFilterFeaturizer,
    LipinskiFilterFeaturizer,
)
from chemcaption.featurize.substructure import IsomorphismFeaturizer
from chemcaption.molecules import Molecule

# Implemented Comparator classes

__all__ = [
    "ValenceElectronCountComparator",
    "LipinskiFilterComparator",
    "GhoseFilterComparator",
    "LeadLikenessFilterComparator",
    "AtomCountComparator",
    "IsomerismComparator",
    "IsomorphismComparator",
    "IsoelectronicComparator",
    "DrugLikenessComparator",
    "BondComparator",
    "MoleculeComparator",
]


class ValenceElectronCountComparator(Comparator):
    """Compare molecular instances for parity based on valence electron count."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(featurizers=[ValenceElectronCountFeaturizer()])

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class LipinskiFilterComparator(Comparator):
    """Compare molecular instances for parity based on number of violations of Lipinski's rule of Five."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(featurizers=[LipinskiFilterFeaturizer()])

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class GhoseFilterComparator(Comparator):
    """Compare molecular instances for parity based on number of violations of the Ghose filter."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(featurizers=[GhoseFilterFeaturizer()])

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class LeadLikenessFilterComparator(Comparator):
    """Compare molecular instances for parity based on number of violations of the lead-likeness drug filter."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(featurizers=[LeadLikenessFilterFeaturizer()])

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class AtomCountComparator(Comparator):
    """Compare molecular instances for atom count."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(featurizers=[AtomCountFeaturizer()])

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class IsomerismComparator(Comparator):
    """Compare molecular instances for parity based on isomerism via molecular formulae."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            featurizers=[
                MolecularFormulaFeaturizer(),
            ]
        )

    def _compare_on_featurizer(
        self,
        featurizer: AbstractFeaturizer,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return results of feature comparison between molecule instances per featurizer.

        Args:
            featurizer (AbstractFeaturizer): Featurizer to compare on.
            molecules (List[Molecule]):
                List containing a pair of molecule instances.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.ndarray: Comparison results. 1 if all extracted features are equal, else 0.
        """
        result = [self.featurizers[0].featurize(molecule).item() for molecule in molecules]
        return np.array([len(set(result)) == 1], dtype=int).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class IsomorphismComparator(Comparator):
    """Compare molecular instances for parity based on isomorphism."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(featurizers=[IsomorphismFeaturizer()])

    def _compare_on_featurizer(
        self,
        featurizer: AbstractFeaturizer,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return results of molecule feature comparison between molecule instance pairs.

        Args:
            featurizer (AbstractFeaturizer): Featurizer to compare on.
            molecules (List[Molecule]):
                List containing a pair of molecule instances.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.ndarray: Comparison results. 1 if all extracted features are equal, else 0.
        """
        result = [self.featurizers[0].featurize(molecule).item() for molecule in molecules]
        return np.array([len(set(result)) == 1], dtype=int).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class IsoelectronicComparator(MultipleComparator):
    """Compare molecular instances for parity based on isoelectronicity."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            comparators=[
                ValenceElectronCountComparator(),
                IsomorphismComparator(),
                AtomCountComparator(),
            ]
        )

    def compare(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """
        Compare for isoelectronic status amongst multiple molecular instances. 1 if all molecules are similar, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.ndarray: Comparison results. 1 if molecules are isoelectronic, else 0.
        """
        return np.reshape(
            self.featurize(molecules=molecules, epsilon=epsilon).all(), (1, 1)
        ).astype(int)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class DrugLikenessComparator(MultipleComparator):
    """Compare molecular instances for similarity based on drug-likeness rules."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            comparators=[
                LipinskiFilterComparator(),
                GhoseFilterComparator(),
                LeadLikenessFilterComparator(),
            ]
        )

    def compare(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """
        Compare multiple molecular instances for drug-likeness status. `1` if all molecules are similar, else `0`.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.ndarray: Comparison results. Array of shape `(1, N)`, where `N` = number of drug-rule comparators.
                Each column from `0` to`N-1` equals `1` if molecules are similar with respect to drug rule, else `0`.
        """
        results = [
            comparator.compare(molecules=molecules, epsilon=epsilon)
            for comparator in self.comparators
        ]
        return np.concatenate(results, axis=1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class BondComparator(Comparator):
    """Compare molecular instances for parity based on intra-molecular bonds."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(featurizers=[BondTypeCountFeaturizer()])

    def _compare_on_featurizer(
        self,
        featurizer: AbstractFeaturizer,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return results of molecule feature comparison between molecule instance pairs.

        Args:
            featurizer (AbstractFeaturizer): Featurizer to compare on.
            molecules (List[Molecule]):
                List containing a pair of molecule instances.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.ndarray: Comparison results. 1 if all extracted features are equal, else 0.
        """
        result = ["_".join(self.featurizers[0]._get_bonds(molecule)) for molecule in molecules]
        return np.array([len(set(result)) == 1], dtype=int).reshape((1, -1))

    def compare(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """
        Compare for bond similarity amongst multiple molecular instances.
            1 if all molecules are identical with respect to bonds, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.ndarray: Comparison results. 1 if molecules are similar with respect to bonds, else 0.
        """
        return np.reshape(
            self.featurize(molecules=molecules, epsilon=epsilon).all(), (1, 1)
        ).astype(int)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MoleculeComparator(Comparator):
    """Compare molecular instances for parity based on molecule identity."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            featurizers=None
        )

    def compare(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """
        Compare for molecule identity status amongst multiple molecular instances. 1 if all molecules are identical, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.ndarray: Comparison results. 1 if molecules are the same, else 0.
        """
        results = np.array([len(set([mol.preprocess_molecule().to_smiles() for mol in molecules])) == 1])
        return results.reshape(1, -1).astype(int)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
