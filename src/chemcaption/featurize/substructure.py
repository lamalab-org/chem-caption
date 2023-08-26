# -*- coding: utf-8 -*-

"""Featurizers describing the structure of (and/or the count and/or presence of substructures in) a molecule."""

from typing import Dict, List, Optional, Union

import numpy as np
import rdkit

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.featurize.text_utils import inspect_info
from chemcaption.molecules import Molecule
from chemcaption.presets import SMARTSPreset

# Implemented molecular structure- and substructure-related featurizers

__all__ = ["SMARTSFeaturizer", "IsomorphismFeaturizer", "TopologyCountFeaturizer"]


"""Featurizer to obtain the presence or count of SMARTS in molecules."""


class SMARTSFeaturizer(AbstractFeaturizer):
    """A featurizer for molecular substructure search via SMARTS."""

    def __init__(
        self,
        count: bool = True,
        names: Optional[Union[str, List[str]]] = "rings",
        smarts: Optional[List[str]] = None,
    ):
        """
        Initialize class.

        Args:
            count (bool): If set to True, count pattern frequency. Otherwise, only encode presence. Defaults to True.
            names (Optional[Union[str, List[str]]]): Preset name(s) of the substructures encoded by the SMARTS strings.
                Predefined presets can be specified as strings, and can be one of:
                    - `heterocyclic`,
                    - `rings`,
                    - `amino`,
                    - `scaffolds`,
                    - `warheads` or
                    - `organic`.
                Defaults to `rings`.
            smarts (Optional[List[str]]): SMARTS strings that are matched with the molecules. Defaults to None.
        """
        super().__init__()

        if isinstance(names, str):
            try:
                self.prefix = f"{names}_"
                names, smarts = SMARTSPreset(names).preset
            except KeyError:
                raise KeyError(
                    f"`{names}` preset not defined. \
                    Use `heterocyclic`, `rings`, 'amino`, `scaffolds`, `warheads`, or `organic`"
                )
        else:
            if bool(names) != bool(smarts):
                raise Exception("Both `names` and `smarts` must either be or not be provided.")

            if len(names) != len(smarts):
                raise Exception("Both `names` and `smarts` must be lists of the same length.")

            self.prefix = "user_provided_"

        self.names = names
        self.smarts = smarts
        self.count = count

        self.suffix = "_count" if count else "_presence"
        self.label = [self.prefix + element.lower() + self.suffix for element in self.names]

        self.template = (
            "What is the "
            + self.suffix[1:]
            + " of the provided {PROPERTY_NAME} (i.e., SMARTS patterns)"
            " in the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": f"functional groups (i.e., {self.prefix[:-1]} preset)",
            }
        ]

    @property
    def preset(self) -> Dict[str, List[str]]:
        """Get molecular preset. Getter method.

        Args:
            None.

        Returns:
            (Dict[str, List[str]]): Dictionary of substance names and substance SMARTS strings.
        """
        return dict(names=self.names, smarts=self.smarts)

    @preset.setter
    def preset(
        self,
        new_preset: Optional[Union[str, Dict[str, List[str]]]],
    ) -> None:
        """Set molecular preset. Setter method.

        Args:
            new_preset (Optional[Union[str, Dict[str, List[str]]]]): New preset of interest.
                Could be a:
                    (str): string representing new predefined preset.
                    (Dict[str, List[str]]): dictionary.
                        Keys: `name` and `smarts`.
                        Values: list of substance names and list of corresponding SMARTS strings.

        Returns:
            None.
        """
        if new_preset is not None:
            if isinstance(new_preset, str):
                names, smarts = SMARTSPreset(preset=new_preset).preset
            elif isinstance(new_preset, (tuple, list)):
                names = new_preset[0]
                smarts = new_preset[1]
            else:
                names = new_preset["names"]
                smarts = new_preset["smarts"]

            self.prefix = f"{new_preset}_" if isinstance(new_preset, str) else "user_provided_"
            self.names = names
            self.smarts = smarts

            self.label = [self.prefix + element.lower() + self.suffix for element in self.names]
        else:
            self.names = None
            self.smarts = None
            self.label = [None]
        return

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
            (np.array): Array containing integer counts/signifier of pattern presence.
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
        """Return feature labels.

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return list(map(lambda x: "".join([("_" if c in "[]()-" else c) for c in x]), self.label))

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
            "According to the Weisfeiler-Lehman isomorphism test, what is the {PROPERTY_NAME} for "
            "the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "Weisfeiler-Lehman graph hash",
            }
        ]

        self.label = ["weisfeiler_lehman_hash"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract and return features from molecular object.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing int representation of isoelectronic status between
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

    def __init__(self, reference_atomic_numbers: Union[int, List[int]] = 6):
        """Initialize class object.

        Args:
            reference_atomic_numbers (Union[int, List[int]]): Atomic number for element of interest.
                Can be either an integer or a list of integers. Defaults to `6` (Carbon).
        """
        super().__init__()

        self.template = "By NMR, what is the {PROPERTY_NAME} for the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        self._names = [
            {
                "noun": "number of unique {} environments",
            }
        ]

        self._reference_atomic_numbers = None
        self.reference_atomic_numbers = reference_atomic_numbers

    @property
    def reference_atomic_numbers(self):
        """Getter for atomic number of `element` of interest.

        Returns:
            atomic_number (int): Atomic number for `element` of interest.
        """
        return self._reference_atomic_numbers

    @reference_atomic_numbers.setter
    def reference_atomic_numbers(self, atomic_number: Union[int, List[int]]):
        """Setter of atomic numbers for `elements` of interest.

        Args:
            atomic_number (Union[int, List[int]]): Atomic numbers for `elements` of interest.

        Returns:
            None.
        """
        self._reference_atomic_numbers = (
            [
                atomic_number,
            ]
            if not isinstance(atomic_number, (list, tuple))
            else atomic_number
        )
        elements = [
            self.periodic_table.GetElementName(atomic_number).lower()
            for atomic_number in self.reference_atomic_numbers
        ]
        self.label = [f"num_{element}_environments" for element in elements]

        info = {"PRECISION": 3, "PROPERTY_VALUE": elements}
        info = inspect_info(info)

        self._names[0]["noun"] = f"number of unique {info['PROPERTY_VALUE']} environments"

        return

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Extract number of unique environments for `elements` of interest.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (np.array): Array containing number of unique `element` environments.
        """
        return np.array(
            [
                self._get_number_of_topologically_distinct_atoms(
                    molecule=molecule, atomic_number=atomic_number
                )
                for atomic_number in self.reference_atomic_numbers
            ]
        ).reshape((1, -1))

    def _get_number_of_topologically_distinct_atoms(
        self, molecule: Molecule, atomic_number: int = 12
    ):
        """Return the number of unique `element` environments based on environmental topology.

        Args:
            molecule (Molecule): Molecular instance.
            atomic_number (int): Atomic number for `element` of interest.

        Returns:
            (int): Number of unique environments.
        """
        mol = rdkit.Chem.AddHs(molecule.rdkit_mol) if atomic_number == 1 else molecule.rdkit_mol

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
