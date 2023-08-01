# -*- coding: utf-8 -*-

"""Implementations for adaptor-related for featurizers."""

from typing import Any, Callable, Dict, List

import numpy as np
from rdkit.Chem import Descriptors

from chemcaption.molecules import Molecule
from chemcaption.featurize.base import AbstractFeaturizer

# Implemented abstract and high-level classes

__all__ = [
    "RDKitAdaptor",  # Higher-level featurizer. Returns lower-level featurizer instances.
    "MolecularMassAdaptor",  # Molar mass featurizer. Subclass of RDKitAdaptor
    "ExactMolecularMassAdaptor",
    "MonoisotopicMolecularMassAdaptor"
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

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Extract and return features from molecular object.

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
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Lower level featurizer adaptors."""

class MolecularMassAdaptor(RDKitAdaptor):
    """Adaptor to extract molar mass information."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(rdkit_function=Descriptors.MolWt, labels=["molecular_mass"])

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Extract and return molecular mass from molecular instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing molecular mass.
        """
        return super().featurize(molecule=molecule)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ExactMolecularMassAdaptor(RDKitAdaptor):
    """Adaptor to extract exact molar mass information."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(rdkit_function=Descriptors.ExactMolWt, labels=["exact_molecular_mass"])

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Extract and return exact molecular mass from molecular instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing exact molecular mass.
        """
        return super().featurize(molecule=molecule)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MonoisotopicMolecularMassAdaptor(RDKitAdaptor):
    """Adaptor to extract monoisotopic molar mass information."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(rdkit_function=Descriptors.ExactMolWt, labels=["monoisotopic_molecular_mass"])

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Extract and return monoisotopic molecular mass from molecular instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing monoisotopic molecular mass.
        """
        return super().featurize(molecule=molecule)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]