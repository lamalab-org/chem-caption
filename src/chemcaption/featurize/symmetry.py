"""Featurizers for symmetry."""
from functools import lru_cache
from typing import List, Union

import numpy as np
from rdkit import Chem

from chemcaption.featurizers import AbstractFeaturizer
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule


@lru_cache(maxsize=128)
def _rdkit_to_pymatgen(mol):
    from givemeconformer.api import get_conformer
    from pymatgen.core import IMolecule  # use immutable for caching

    from .utils import get_atom_symbols_and_positions

    c = get_conformer(Chem.MolToSmiles(mol))[0]
    m = IMolecule(*get_atom_symbols_and_positions(c))
    return m


@lru_cache(maxsize=128)
def _pmg_mol_to_pointgroup_analyzer(mol):
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    analyzer = PointGroupAnalyzer(mol)
    return analyzer


class RotationalSymmetryNumber(AbstractFeaturizer):
    """Obtain the rotational symmetry number of a molecule."""

    def __init__(self):
        """Initialize instance."""
        self.template = "What is the roitational symmetry number for a molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        self._names = [
            {
                "noun": "rotational symmetry number",
            }
        ]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance.
        Returns the rotational symmetry number of a molecule.

        The symmetry number is the number of indistinguishable rotated positions.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: Rotational symmetry number.
        """
        mol = molecule.rdkit_mol
        m = _rdkit_to_pymatgen(mol)
        analyzer = _pmg_mol_to_pointgroup_analyzer(m)
        return np.array([analyzer.get_rotational_symmetry_number()])

    def feature_labels(self) -> List[str]:
        return ["rotational_symmetry_number"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Kevin Maik Jablonka"]


class PointGroupFeaturizer(AbstractFeaturizer):
    """Return point group of molecule."""

    def __init__(self):
        self.template = "What is the point group of a molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        self._names = [
            {
                "noun": "point group",
            }
        ]

    # ToDo: consider if we want to continue
    # returning the point group as a string
    # I think we have to, because there are infinitely many
    # and for one-hot encoding we would need to know the
    # possible point groups beforehand
    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance.
        Returns the point group of a molecule.

        Note that infinity is represented as * in the output.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: Schoenflies symbol of point group.
        """
        mol = molecule.rdkit_mol
        m = _rdkit_to_pymatgen(mol)
        analyzer = _pmg_mol_to_pointgroup_analyzer(m)
        return np.array([analyzer.get_pointgroup().sch_symbol])

    def feature_labels(self) -> List[str]:
        return ["point_group"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Kevin Maik Jablonka"]
