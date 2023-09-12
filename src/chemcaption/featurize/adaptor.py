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
    "MolecularMassAdaptor",  # Molar mass featurizer. Subclass of RDKitAdaptor
    "MonoisotopicMolecularMassAdaptor",  # Monoisotopic molar mass featurizer
    "HydrogenDonorCountAdaptor",  # Hydrogen donor count featurizer
    "HydrogenAcceptorCountAdaptor",  # Hydrogen acceptor count featurizer
    "RotableBondCountAdaptor",  # Rotatable bond counter (non-strict)
    "NonRotableBondCountAdaptor",  # Non-rotatable bond counter (non-strict)
    "StrictRotableBondCountAdaptor",  # Rotatable bond counter (strict)
    "StrictNonRotableBondCountAdaptor",  # Non-rotatable bond counter (strict)
    "ValenceElectronCountAdaptor",
    "RotableBondDistributionAdaptor",
    "StrictRotableBondDistributionAdaptor",
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

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return self._label

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

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

        self.template = (
            "What is the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "molecular mass",
            }
        ]

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
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MonoisotopicMolecularMassAdaptor(RDKitAdaptor):
    """Adaptor to extract monoisotopic molar mass information."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            rdkit_function=Descriptors.ExactMolWt, labels=["monoisotopic_molecular_mass"]
        )

        self.template = (
            "What is the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "monoisotopic molecular mass",
            }
        ]

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
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class HydrogenDonorCountAdaptor(RDKitAdaptor):
    """Adaptor to extract number of Hydrogen bond donors."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(rdkit_function=Descriptors.NumHDonors, labels=["num_hydrogen_bond_donors"])

        self.template = (
            "What is the {PROPERTY_NAME} in the molecule" " with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "number of hydrogen bond acceptors",
            }
        ]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Extract and return the number of Hydrogen bond donors in molecular instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing number of Hydrogen bond donors.
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


class HydrogenAcceptorCountAdaptor(RDKitAdaptor):
    """Adaptor to extract number of Hydrogen bond acceptors."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            rdkit_function=Descriptors.NumHAcceptors, labels=["num_hydrogen_bond_acceptors"]
        )

        self.template = (
            "What is the {PROPERTY_NAME} in the molecule" " with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "number of hydrogen bond acceptors",
            }
        ]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance.

        Extract and return the number of Hydrogen bond acceptors in molecular instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing number of Hydrogen bond acceptors.
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


class RotableBondCountAdaptor(RDKitAdaptor):
    """Adaptor to extract number of rotatable bonds (non-strict criterion)."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            rdkit_function=rdMolDescriptors.CalcNumRotatableBonds,
            labels=["num_rotable_bonds"],
            **{"strict": False},
        )

        self.template = (
            "What is the {PROPERTY_NAME} in the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "number of rotatable bonds",
            }
        ]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Extract and return the number of rotable bonds in molecular instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing number of rotable bonds in molecule.
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


class StrictRotableBondCountAdaptor(RDKitAdaptor):
    """Adaptor to extract number of rotatable bonds (strict criterion)."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            rdkit_function=rdMolDescriptors.CalcNumRotatableBonds,
            labels=["num_rotable_bonds_strict"],
            **{"strict": True},
        )

        self.template = (
            "What is the {PROPERTY_NAME} in the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "number of rotatable bonds (by strict guidelines)",
            }
        ]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance.

        Extract and return the number of rotable bonds in molecular instance (strict criterion).

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing number of strictly rotable bonds.
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


class NonRotableBondCountAdaptor(RDKitAdaptor):
    """Adaptor to extract number of rotatable bonds (non-strict criterion)."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            rdkit_function=rdMolDescriptors.CalcNumRotatableBonds,
            labels=["num_non_rotable_bonds"],
            **{"strict": False},
        )

        self.template = (
            "What is the {PROPERTY_NAME} in the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "number of non-rotatable bonds",
            }
        ]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Extract and return the number of non-rotatable bonds in molecular instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing number of non-rotatable bonds in molecule.
        """
        num_bonds = len(molecule.reveal_hydrogens().GetBonds())
        return num_bonds - super().featurize(molecule=molecule)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class StrictNonRotableBondCountAdaptor(RDKitAdaptor):
    """Adaptor to extract number of non-rotatable bonds (strict criterion)."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            rdkit_function=rdMolDescriptors.CalcNumRotatableBonds,
            labels=["num_non_rotable_bonds_strict"],
            **{"strict": True},
        )

        self.template = (
            "What is the {PROPERTY_NAME} in the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "number of non-rotatable bonds (by strict guidelines)",
            }
        ]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance.

        Extract and return the number of non-rotatable bonds in molecular instance (strict criterion).

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing number of strictly non-rotatable bonds.
        """
        num_bonds = len(molecule.reveal_hydrogens().GetBonds())
        return num_bonds - super().featurize(molecule=molecule)

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

        self.template = (
            "What is the {PROPERTY_NAME} for the molecule"
            " with {REPR_SYSTEM} `{REPR_STRING}` have in its outer shell?"
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


class RotableBondDistributionAdaptor(RDKitAdaptor):
    """Adaptor to extract distribution of rotatable and non-rotatable bonds (non-strict criterion)."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            rdkit_function=rdMolDescriptors.CalcNumRotatableBonds,
            labels=["rotable_proportion", "non_rotable_proportion"],
            **{"strict": False},
        )

        self.template = (
            "What is the {PROPERTY_NAME} in the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "proportion of rotatable and non-rotatable bonds",
            }
        ]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Extract and return the proportion of rotatable and non-rotatable bonds
            in molecular instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing proportion of rotatable and non-rotatable bonds in molecule.
        """
        num_bonds = len(molecule.reveal_hydrogens().GetBonds())
        num_rotable = super().featurize(molecule=molecule)
        return np.array([num_rotable / num_bonds, (num_bonds - num_rotable) / num_bonds]).reshape(
            (1, -1)
        )

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class StrictRotableBondDistributionAdaptor(RDKitAdaptor):
    """Adaptor to extract number of rotatable bonds (strict criterion)."""

    def __init__(self):
        """Initialize instance."""
        super().__init__(
            rdkit_function=rdMolDescriptors.CalcNumRotatableBonds,
            labels=["rotable_proportion_strict", "non_rotable_proportion_strict"],
            **{"strict": True},
        )

        self.template = (
            "What is the {PROPERTY_NAME} in the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "proportion of rotatable and non-rotatable bonds (by strict guidelines)",
            }
        ]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance.

        Extract and return the distribution of rotatable and non-rotatable bonds in
            molecular instance based on the strict criterion.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing distribution of strictly rotatable and non-rotatable bonds.
        """
        num_bonds = len(molecule.reveal_hydrogens().GetBonds())
        num_rotable = super().featurize(molecule=molecule)
        return np.array([num_rotable / num_bonds, (num_bonds - num_rotable) / num_bonds]).reshape(
            (1, -1)
        )

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
