# -*- coding: utf-8 -*-

"""Comparator implementations."""

from typing import List

import numpy as np

from chemcaption.featurize.base import AbstractFeaturizer, Comparator, MultipleComparator
from chemcaption.featurize.composition import AtomCountFeaturizer, MolecularFormulaFeaturizer
from chemcaption.featurize.electronicity import ValenceElectronCountFeaturizer
from chemcaption.featurize.rules import LipinskiViolationCountFeaturizer
from chemcaption.featurize.substructure import IsomorphismFeaturizer
from chemcaption.molecules import Molecule

# Implemented Comparator classes

__all__ = [
    "ValenceElectronCountComparator",
    "LipinskiViolationCountComparator",
    "AtomCountComparator",
    "IsomerismComparator",
    "IsomorphismComparator",
    "IsoelectronicComparator",
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


class LipinskiViolationCountComparator(Comparator):
    """Compare molecular instances for parity based on number of violations of Lipinski's rule of Five."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(featurizers=[LipinskiViolationCountFeaturizer()])

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
    ) -> np.array:
        """Return results of feature comparison between molecule instances per featurizer.

        Args:
            featurizer (AbstractFeaturizer): Featurizer to compare on.
            molecules (List[Molecule]):
                List containing a pair of molecule instances.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.array: Comparison results. 1 if all extracted features are equal, else 0.
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
    ) -> np.array:
        """Return results of molecule feature comparison between molecule instance pairs.

        Args:
            featurizer (AbstractFeaturizer): Featurizer to compare on.
            molecules (List[Molecule]):
                List containing a pair of molecule instances.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.array: Comparison results. 1 if all extracted features are equal, else 0.
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
    ) -> np.array:
        """
        Compare for isoelectronic status amongst multiple molecular instances. 1 if all molecules are similar, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            np.array: Comparison results. 1 if molecules are isoelectronic, else 0.
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
