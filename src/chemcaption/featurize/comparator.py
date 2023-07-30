# -*- coding: utf-8 -*-

"""Comparator implementations"""

from typing import List, Union

import numpy as np

from chemcaption.featurize.base import AbstractFeaturizer, Comparator, MultipleComparator
from chemcaption.featurize.composition import AtomCountFeaturizer
from chemcaption.featurize.electronicity import ValenceElectronCountFeaturizer
from chemcaption.featurize.substructure import IsomorphismFeaturizer
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

# Implemented Comparator classes

__all__ = [
    "ValenceElectronCountComparator",
    "AtomCountComparator",
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
            None

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
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class IsomorphismComparator(Comparator):
    """Compare molecular instances for isomorphism."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(featurizers=[IsomorphismFeaturizer()])

    def _compare_on_featurizer(
        self,
        featurizer: AbstractFeaturizer,
        molecules: List[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]],
        epsilon: float = 0.0,
    ) -> np.array:
        """Return results of molecule feature comparison between molecule instance pairs.

        Args:
            featurizer (AbstractFeaturizer): Featurizer to compare on.
            molecules (List[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]]):
                List containing a pair of molecule instances.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            (np.array): Comparison results. 1 if all extracted features are equal, else 0.
        """
        result = [self.featurizers[0].featurize(molecule) for molecule in molecules]
        return np.array([len(set(result)) == 1], dtype=int).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class IsoelectronicComparator(MultipleComparator):
    """Compare molecular instances for isoelectronicity."""

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
        molecules: List[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]],
        epsilon: float = 0.0,
    ) -> np.array:
        return self.featurize(molecules=molecules, epsilon=epsilon).astype(int).all()

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
