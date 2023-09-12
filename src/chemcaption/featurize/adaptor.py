# -*- coding: utf-8 -*-

"""Implementations for adaptor-related for featurizers."""

from typing import Any, Callable, Dict, List

import numpy as np
from rdkit.Chem import Descriptors, rdMolDescriptors

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import Molecule

# Implemented high- and low-level adaptor classes

__all__ = [
    "RDKitAdaptor",  # Higher-level featurizer. Returns lower-level featurizer instances.
    "ValenceElectronCountAdaptor",  # Adaptor to extract for valence electron count.
]


"""High-level featurizer adaptor."""


class RDKitAdaptor(AbstractFeaturizer):
    """Higher-level featurizer. Returns specific, lower-level featurizers."""

    def __init__(
        self, rdkit_function: Callable, labels: List[str], **rdkit_function_kwargs: Dict[str, Any]
    ):
        """Initialize class object.

        Args:
            rdkit_function (Callable): Molecule descriptor-generating function.
                May be obtained from a chemistry featurization package like `rdkit` or custom written.
            labels (List[str]): Feature label(s) to assign to extracted feature(s).
            rdkit_function_kwargs (Dict[str, Any]): Keyword arguments to be parsed by `rdkit_function`.
        """
        super().__init__()
        self.rdkit_function = rdkit_function
        self._label = labels
        self.rdkit_function_kwargs = rdkit_function_kwargs

    def feature_labels(self) -> List[str]:
        return self._label

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance.

        Extract and return features from molecular object.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing extracted features.
        """
        feature = self.rdkit_function(molecule.rdkit_mol, **self.rdkit_function_kwargs)
        feature = (
            [
                feature,
            ]
            if isinstance(feature, (int, float))
            else feature
        )
        return np.array(feature).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ValenceElectronCountAdaptor(RDKitAdaptor):
    """Adaptor to extract for valence electron count."""

    def __init__(self):
        """Initialize class.

        Args:
            None
        """
        super().__init__(
            rdkit_function=Descriptors.NumValenceElectrons,
            labels=["num_valence_electrons"],
        )

        self._names = [
            {
                "noun": "number of valence electrons",
            },
            {
                "noun": "valence electron count",
            },
            {
                "noun": "count of valence electrons",
            },
        ]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract and return valence electron count for molecular object.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing number of valence electrons.
        """
        return super().featurize(molecule=molecule)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
