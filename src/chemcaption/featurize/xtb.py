# -*- coding: utf-8 -*-

"""Featurizers based on xTB properties."""

from typing import List, Dict, Any, Optional
import os

import numpy as np

from rdkit import Chem

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import Molecule, SMILESMolecule
from chemcaption.featurize.utils import join_list_elements, cached_conformer

from frozendict import frozendict

from morfeus import read_xyz, XTB


# Implemented featurizers

__all__ = [
    "XTBFeaturizer",
    "ElectronAffinityFeaturizer",
]

class XTBFeaturizer(AbstractFeaturizer):
    """Abstract featurizer for XTB features."""

    def __init__(self, file_name: Optional[str] = None, conformer_generation_kwargs: Optional[Dict[str, Any]] = None):
        """Instantiate class.

        Args:
            file_name (Optional[str]): Name for temporary XYZ file.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
        """
        super().__init__()
        self._conf_gen_kwargs = (
            frozendict(conformer_generation_kwargs)
            if conformer_generation_kwargs
            else frozendict({})
        )

        if file_name is None:
            num = np.random.randint(low=1, high=100)
            numbers = np.random.randint(low=65, high=82, size=(num,)).flatten().tolist()
            letters = list(map(lambda x: chr(x), numbers))

            file_name = "".join(letters) + ".xyz"

        self.random_file_name = file_name if file_name.endswith(".xyz") else file_name + ".xyz"

    def _get_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        """Return conformer for molecule.

        Args:
            mol (Chem.Mol): rdkit Molecule.

        Returns:
            (Chem.Mol): Molecule instance embedded with conformers.
        """
        smiles = Chem.MolToSmiles(mol)
        return cached_conformer(smiles, self._conf_gen_kwargs)

    def _mol_to_xyz_file(self, molecule: Molecule) -> None:
        """Generate XYZ block from molecule instance.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            None.
        """
        mol = molecule.rdkit_mol
        mol = self._get_conformer(mol)

        Chem.rdmolfiles.MolToXYZFile(mol, self.random_file_name)
        return

    def _xyz_file_to_mol(self) -> SMILESMolecule:
        """Generate XYZ block from molecule instance.

        Args:
            None.

        Returns:
            (SMILESMolecule): SMILES molecule instance.
        """
        mol = Chem.rdmolfiles.MolFromXYZFile(self.random_file_name)
        smiles = Chem.MolToSmiles(mol)

        return SMILESMolecule(smiles)

    def _get_xtb_instance(self, molecule: Molecule) -> XTB:
        """Return XTB instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (XTB): XTB instance.
        """
        self._mol_to_xyz_file(molecule) # Persist molecule in XYZ file
        elements, coordinates = read_xyz(self.random_file_name) # Read file

        os.remove(self.random_file_name) # Eliminate file

        return XTB(elements, coordinates, "1")

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ElectronAffinityFeaturizer(XTBFeaturizer):
    """Featurize molecule and return electron affinity."""

    def __init__(self, file_name: Optional[str] = None, conformer_generation_kwargs: Optional[Dict[str, Any]] = None):
        """Instantiate class.

        Args:
            file_name (Optional[str]): Name for temporary XYZ file.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
        """
        super().__init__(file_name=file_name, conformer_generation_kwargs=conformer_generation_kwargs)

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing electron affinity for molecule instance.
        """
        xtb = super()._get_xtb_instance(molecule=molecule)
        return xtb.get_ea()

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return ["electron_affinity"]
