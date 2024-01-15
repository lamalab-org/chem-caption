# -*- coding: utf-8 -*-

"""Featurizers describing the structure of (and/or the count and/or presence of substructures in) a molecule."""

from typing import Dict, List, Optional

import numpy as np
import rdkit
from rdkit.Chem import GetPeriodicTable, PeriodicTable

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.featurize.utils import join_list_elements
from chemcaption.molecules import Molecule
from chemcaption.presets import SMARTS_MAP

__all__ = ["SMARTSFeaturizer", "IsomorphismFeaturizer", "TopologyCountFeaturizer"]


"""Featurizer to obtain the presence or count of SMARTS in molecules."""


class SMARTSFeaturizer(AbstractFeaturizer):
    """A featurizer for molecular substructure search via SMARTS."""

    def __init__(
        self,
        smarts: List[str],
        names: Optional[List[str]],
        count: bool = True,
    ):
        """
        Initialize class.

        Args:
            smarts (Optional[List[str]]): SMARTS strings that are matched with the molecules.
                Defaults to `None`.
            names (Optional[List[str]]): Names of the SMARTS strings.
                If `None`, the SMARTS strings are used as names.
                Defaults to `None`.
            count (bool): If set to `True`, count pattern frequency.
                Otherwise, only encode presence.
                Defaults to `True`.
        """
        super().__init__()

        self.smart_names = names if names is not None else smarts
        self.smarts = smarts
        self.count = count
        self.constraint = (
            "Constraint: return a list of integers."
            if self.count
            else "Constraint: return a list of 1s and 0s if the pattern is present or not."
        )

        self.prompt_template = "{PROPERTY_NAME} in the molecule with {REPR_SYSTEM} {REPR_STRING}?"

    def get_names(self) -> List[Dict[str, str]]:
        """Return names of extracted features.

        Args:
            None.

        Returns:
            List[Dict[str, str]]: List of dictionaries containing feature names.
        """
        if len(self.smart_names) == 1:
            name = "Is"
            noun = "count"
        else:
            name = "Are"
            noun = "counts"

        if self.count:
            name = f"Question: What {name.lower()} the {noun} of " + join_list_elements(self.smart_names)

        else:
            name = f"Question: {name} " + join_list_elements(self.smart_names)

        return [{"noun": name}]

    @classmethod
    def from_preset(cls, preset: str, count: bool = True):
        """
        Args:
            preset (str): Preset name of the substructures
                encoded by the SMARTS strings.
                Predefined presets can be specified as strings, and can be one of:
                    - `heterocyclic`,
                    - `rings`,
                    - `amino`,
                    - `scaffolds`,
                    - `warheads` or
                    - `organic`.
                    - `all`
            count (bool): If set to True, count pattern frequency.
        """
        if preset not in SMARTS_MAP:
            raise ValueError(
                f"Invalid preset name '{preset}'. "
                f"Valid preset names are: {', '.join(SMARTS_MAP.keys())}."
            )
        smarts_set = SMARTS_MAP[preset]
        return cls(smarts=smarts_set["smarts"], names=smarts_set["names"], count=count)

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance.

        Return integer array representing the:
            - frequency or
            - presence
            of molecular patterns in a molecule.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.array: Array containing integer counts/signifier of pattern presence.
        """
        if self.count:
            results = [
                len(molecule.rdkit_mol.GetSubstructMatches(rdkit.Chem.MolFromSmarts(smart)))
                for smart in self.smarts
            ]
        else:
            results = [
                int(molecule.rdkit_mol.HasSubstructMatch(rdkit.Chem.MolFromSmarts(smart)))
                for smart in self.smarts
            ]

        return np.array(results).reshape((1, -1))

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels of extracted features.
        """
        suffix = "_count" if self.count else "_presence"
        return [name + suffix for name in self.smart_names]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class IsomorphismFeaturizer(AbstractFeaturizer):
    """Convert molecule graph to Weisfeiler-Lehman hash."""

    def __init__(self):
        """Instantiate class."""
        super().__init__()

        self.template = (
            "According to the Weisfeiler-Lehman isomorphism test, what {VERB} the {PROPERTY_NAME} for "
            "the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "Weisfeiler-Lehman graph hash",
            }
        ]

        self.label = ["weisfeiler_lehman_hash"]

    def feature_labels(self) -> List[str]:
        return ["weisfeiler_lehman_hash"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract and return features from molecular object.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.array: Array containing int representation of isoelectronic status between
                `self.reference_molecule` and `molecule`.
        """
        molecule_graph = molecule.to_graph()

        return np.array(molecule_graph.weisfeiler_lehman_graph_hash()).reshape(1, 1)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class TopologyCountFeaturizer(AbstractFeaturizer):
    """Featurizer to return number of unique `element` environments in a molecule."""

    def __init__(self, reference_atomic_numbers: List[int]):
        """Initialize class object.

        Args:
            reference_atomic_numbers (List[int]): Atomic number(s) for element(s) of interest.
        """
        super().__init__()
        self.reference_atomic_numbers = reference_atomic_numbers

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels for extracted features.
        """
        return [
            "topology_count_" + str(atomic_number)
            for atomic_number in self.reference_atomic_numbers
        ]

    def get_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            List[Dict[str, str]]: List of names for extracted features according to parts-of-speech.
        """
        # map the numbers to names
        periodic_table = GetPeriodicTable()
        names = [
            PeriodicTable.GetElementSymbol(periodic_table, atomic_number)
            for atomic_number in self.reference_atomic_numbers
        ]

        noun = "numbers" if len(self.reference_atomic_numbers) > 1 else "number"
        return [
            {"noun": f"{noun} of topologically unique environments of {join_list_elements(names)}"}
        ]

    @classmethod
    def from_preset(cls, preset: str):
        """Generate class instance with atomic numbers of interest based on predefined presets.

        Args:
            preset (str): Preset of interest.

        Returns:
            self: Instance of self.
        """
        if preset == "organic":
            # Use C, H, N, O, P, S, F, Cl, Br, I
            return cls(reference_atomic_numbers=[6, 1, 7, 8, 15, 16, 9, 17, 35, 53])
        elif preset == "carbon":
            return cls(reference_atomic_numbers=[6])
        elif preset == "nitrogen":
            return cls(reference_atomic_numbers=[7])

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract number of unique environments for `elements` of interest.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.array: Array containing number of unique `element` environments.
        """
        return np.array(
            [
                self._get_number_of_topologically_distinct_atoms(
                    molecule=molecule, atomic_number=atomic_number
                )
                for atomic_number in self.reference_atomic_numbers
            ]
        ).reshape((1, -1))

    @staticmethod
    def _get_number_of_topologically_distinct_atoms(molecule: Molecule, atomic_number: int = 12):
        """Return the number of unique `element` environments based on environmental topology.

        Args:
            molecule (Molecule): Molecular instance.
            atomic_number (int): Atomic number for `element` of interest.

        Returns:
            int: Number of unique environments.
        """
        mol = molecule.reveal_hydrogens() if atomic_number == 1 else molecule.rdkit_mol

        # Get unique canonical atom rankings
        atom_ranks = list(rdkit.Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))

        # Select the unique element environments
        atom_ranks = np.array(atom_ranks)

        # Atom indices
        atom_indices = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_number
        ]
        # Count them
        return len(set(atom_ranks[atom_indices]))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]
