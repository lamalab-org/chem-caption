# -*- coding: utf-8 -*-

"""Featurizers for electron-related information."""
from

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

    def featurize(self, molecule: Union[SMILESMolecule, SELFIESMolecule, InChIMolecule])-> np.array:
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
        self.label = ["isoelectronic_similarity"]
        self.comparer = ValenceElectronCountFeaturizer()

    def featurize(self, molecule: Union[SMILESMolecule, SELFIESMolecule, InChIMolecule])-> np.array:
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

    def featurize(self, molecule: Union[SMILESMolecule, SELFIESMolecule, InChIMolecule])-> np.array:
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

        return np.array([num_reference_valence_electrons == num_valence_electrons], dtype=int).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


if __name__ == "__main__":
    mol1 = SELFIESMolecule("CC(=O)NCCCOc1cccc(CN2CCCCC2)c1")
    mol2 = SELFIESMolecule("[C][C][Branch1][C][C][N][C][C][Branch1][C][O][C][O][C][=C][C][=C][C][=C][C][=C][C][=C][Ring1][#Branch2][Ring1][=Branch1].[ClH0]")

    feat1 = ValenceElectronCountFeaturizer()
    feat2 = IsoelectronicityFeaturizer(mol1)

    print(feat1.featurize(mol1))
    print(feat1.featurize(mol2))

    print(feat2.featurize(mol2))