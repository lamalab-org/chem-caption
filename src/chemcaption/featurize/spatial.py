# -*- coding: utf-8 -*-

"""Featurizers for chemical bond-related information."""

from typing import List, Optional

import numpy as np
from rdkit.Chem import Descriptors3D, rdMolDescriptors
from rdkit.Chem.AllChem import EmbedMolecule

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import Molecule

# Implemented bond-related featurizers

__all__ = [
    "EccentricityFeaturizer",
]


class ThreeDimensionalFeaturizer(AbstractFeaturizer):

    def __init__(self, conformer_id: Optional[int] = -1, use_masses: bool = True, force=True):
        super().__init__()

        self.conformer_id = conformer_id
        self.use_masses = use_masses
        self.force = force


class EccentricityFeaturizer(ThreeDimensionalFeaturizer):
    """Featurizer to return number of unique `element` environments in a molecule."""

    def __init__(self, conformer_id: Optional[int] = -1, use_masses: bool = True, force=True):
        """Initialize class object.

        Args:
            conformer_id (Optional[int]): Integer identifier for molecule conformation. Defaults to `-1`.
            use_masses (bool): Utilize elemental masses in asphericity calculation. Defaults to `True`.
            force (bool):
        """
        super().__init__(conformer_id=conformer_id, use_masses=use_masses, force=force)

        self.label = ["eccentricity"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract eccentricity value for `molecule`.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing eccentricity value.
        """
        mol = molecule.reveal_hydrogens()
        _ = EmbedMolecule(mol)
        eccentricity_value = Descriptors3D.Eccentricity(
            mol
        )
        return np.array([eccentricity_value]).reshape(1, -1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
