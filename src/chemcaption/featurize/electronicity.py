# -*- coding: utf-8 -*-

"""Featurizers for proton- and electron-related information."""

from typing import List, Union

import numpy as np
from rdkit.Chem import Descriptors, rdMolDescriptors

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

# Implemented proton-, electron- and charge-related featurizers

__all__ = [
    "HydrogenAcceptorCountFeaturizer",
    "HydrogenDonorCountFeaturizer",
    "ValenceElectronCountFeaturizer",
    "IsoelectronicDifferenceFeaturizer",
    "IsoelectronicityFeaturizer",
]


"""Featurizer to extract hydrogen acceptor count from molecules."""

class HydrogenAcceptorCountFeaturizer(AbstractFeaturizer):
    """Obtain number of Hydrogen bond acceptors in a molecule."""

    def __init__(self):
        """Get the number of Hydrogen bond acceptors present in a molecule."""
        super().__init__()
        self.label = ["num_hydrogen_bond_acceptors"]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance. Get the number of Hydrogen bond acceptors present in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            (np.array): Number of Hydrogen bond acceptors present in `molecule`.
        """
        return np.array([rdMolDescriptors.CalcNumHBA(molecule.rdkit_mol)]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer to extract hydrogen donor count from molecules."""

class HydrogenDonorCountFeaturizer(AbstractFeaturizer):
    """Obtain number of Hydrogen bond donors in a molecule."""

    def __init__(self):
        """Get the number of Hydrogen bond donors present in a molecule."""
        super().__init__()
        self.label = ["num_hydrogen_bond_donors"]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance. Get the number of Hydrogen bond donors present in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: Number of Hydrogen bond donors present in `molecule`.
        """
        return np.array([rdMolDescriptors.CalcNumHBD(molecule.rdkit_mol)]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer to obtain molecular valence electron count"""

class ValenceElectronCountFeaturizer(AbstractFeaturizer):
    """A featurizer for molecular electronicity-based comparison."""

    def __init__(self):
        """Initialize class.

        Args:
            None
        """
        super().__init__()
        self.label = ["num_valence_electrons"]

    def featurize(
        self, molecule: Union[SMILESMolecule, SELFIESMolecule, InChIMolecule]
    ) -> np.array:
        """
        Extract and return valence electron count for molecular object.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

        Returns:
            (np.array): Array containing number of valence electrons.
        """
        num_valence_electrons = Descriptors.NumValenceElectrons(molecule.reveal_hydrogens())

        return np.array([num_valence_electrons]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer to compare molecules for isoelectronic difference"""

class IsoelectronicDifferenceFeaturizer(AbstractFeaturizer):
    """A featurizer for molecular electronicity-based comparison."""

    def __init__(self, reference_molecule: Union[SMILESMolecule, SELFIESMolecule, InChIMolecule]):
        """Initialize class.

        Args:
            reference_molecule (Union[SMILESMolecule, SELFIESMolecule, InChIMolecule]): Molecule representation.
        """
        super().__init__()
        self.reference_molecule = reference_molecule
        self.label = ["isoelectronic_difference"]
        self.comparer = ValenceElectronCountFeaturizer()

    def featurize(
        self, molecule: Union[SMILESMolecule, SELFIESMolecule, InChIMolecule]
    ) -> np.array:
        """
        Extract and return features from molecular object.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

        Returns:
            (np.array): Array containing int representation of isoelectronic status between
                `self.reference_molecule` and `molecule`.
        """
        num_valence_electrons = self.comparer.featurize(molecule)
        num_reference_valence_electrons = self.comparer.featurize(self.reference_molecule)

        return np.array([num_reference_valence_electrons - num_valence_electrons]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


"""Featurizer to compare molecules for isoelectronicity"""

class IsoelectronicityFeaturizer(AbstractFeaturizer):
    """A featurizer for molecular electronicity-based comparison."""

    def __init__(self, reference_molecule: Union[SMILESMolecule, SELFIESMolecule, InChIMolecule]):
        """Initialize class.

        Args:
            reference_molecule (Union[SMILESMolecule, SELFIESMolecule, InChIMolecule]): Molecule representation.
        """
        super().__init__()
        self.reference_molecule = reference_molecule
        self.label = ["isoelectronic_similarity"]
        self.comparer = ValenceElectronCountFeaturizer()

    def featurize(
        self, molecule: Union[SMILESMolecule, SELFIESMolecule, InChIMolecule]
    ) -> np.array:
        """
        Extract and return features from molecular object.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

        Returns:
            (np.array): Array containing int representation of isoelectronic status between
                `self.reference_molecule` and `molecule`.
        """
        num_valence_electrons = self.comparer.featurize(molecule)
        num_reference_valence_electrons = self.comparer.featurize(self.reference_molecule)

        return np.array(
            [num_reference_valence_electrons == num_valence_electrons], dtype=int
        ).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
