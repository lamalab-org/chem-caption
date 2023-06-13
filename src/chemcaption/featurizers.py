# -*- coding: utf-8 -*-

"""Utility imports."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import rdkit
from rdkit.Chem import Lipinski, rdMolDescriptors

from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule
from chemcaption.presets import SMARTSPreset

"""Abstract classes."""


class AbstractFeaturizer(ABC):
    """Base class for lower level Featurizers.

    Args:
        None

    Returns:
        None
    """

    def __init__(self):
        """Initialize class. Initialize periodic table."""
        self.periodic_table = rdkit.Chem.GetPeriodicTable()
        self._label = list()

    @abstractmethod
    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """Featurize single Molecule instance."""
        raise NotImplementedError

    def featurize_many(
        self, molecules: Sequence[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]]
    ) -> np.array:
        """
        Featurize a sequence of Molecule objects.

        Args:
            molecules (Sequence[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]]):
                A sequence of molecule representations.

        Returns:
            np.array: An array of features for each molecule instance.
        """
        return np.concatenate([self.featurize(molecule) for molecule in molecules])

    def text_featurize(self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]):
        """Embed features in Prompt instance."""
        return None

    def text_featurize_many(self, molecules: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]):
        """Embed features in Prompt instance for multiple molecules."""
        return None

    @abstractmethod
    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        raise NotImplementedError

    @property
    def label(self) -> List[str]:
        """Get label attribute. Getter method."""
        return self._label

    @label.setter
    def label(self, new_label: List[str]) -> None:
        """Set label attribute. Setter method. Changes instance state.

        Args:
            new_label (str): New label for generated feature.

        Returns:
            None
        """
        self._label = new_label
        return

    def feature_labels(self) -> List[str]:
        """Return feature label.

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
        """
        return self.label

    def citations(self):
        """Return citation for this project."""
        return None


"""
Lower level featurizer classes.

1. NumRotableBondsFeaturizer []
2. BondRotabilityFeaturizer []
3. HAcceptorCountFeaturizer []
4. HDonorCountFeaturizer []
5. MolecularMassFeaturizer []
6. ElementMassFeaturizer []
7. ElementCountFeaturizer []
8. ElementMassProportionFeaturizer []
9. ElementCountProportionFeaturizer []
10. MultipleFeaturizer
11. SMARTSFeaturizer
"""


class NumRotableBondsFeaturizer(AbstractFeaturizer):
    """Obtain number of rotable (i.e., non-terminal, non-hydrogen) bonds in a molecule."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()
        self.label = ["num_rotable_bonds"]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Count the number of rotable (single, non-terminal) bonds in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

        Returns:
            num_rotable (np.array): Number of rotable bonds in molecule.
        """
        num_rotable = rdMolDescriptors.CalcNumRotatableBonds(molecule.rdkit_mol, strict=True)
        return np.array([num_rotable]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class BondRotabilityFeaturizer(AbstractFeaturizer):
    """Obtain distribution between rotable and non-rotable bonds in a molecule."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()
        self.label = ["rotable_proportion", "non_rotable_proportion"]

    def _get_bond_types(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> List[float]:
        """Return distribution of bonds based on rotability.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            bond_distribution (List[float]): Distribution of bonds based on rotability.
        """
        num_bonds = len(molecule.rdkit_mol.GetBonds())
        num_rotable = rdMolDescriptors.CalcNumRotatableBonds(molecule.rdkit_mol, strict=False)
        num_non_rotable = num_bonds - num_rotable

        bond_distribution = [num_rotable / num_bonds, num_non_rotable / num_bonds]

        return bond_distribution

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """Featurize single molecule instance. Return distribution of bonds based on rotability.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: Array containing distribution of the bonds based on rotability.
        """
        return np.array(self._get_bond_types(molecule=molecule)).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class HAcceptorCountFeaturizer(AbstractFeaturizer):
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


class HDonorCountFeaturizer(AbstractFeaturizer):
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


class MolecularMassFeaturizer(AbstractFeaturizer):
    """Get the molecular mass of a molecule."""

    def __init__(self):
        """Get the molecular mass of a molecule."""
        super().__init__()
        self.label = ["molecular_mass"]

    def featurize(
        self,
        molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule],
    ) -> np.array:
        """
        Featurize single molecule instance. Get the molecular mass of a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            molar_mass (float): Molecular mass of `molecule`.
        """
        molar_mass = sum(
            [
                self.periodic_table.GetAtomicWeight(atom.GetAtomicNum())
                for atom in molecule.get_atoms()
            ]
        )
        return np.array([molar_mass]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ElementMassFeaturizer(AbstractFeaturizer):
    """Obtain mass for elements in a molecule."""

    def __init__(self, preset: Optional[Union[List[str], Dict[str, str]]] = None):
        """Get the total mass component of an element in a molecule.

        Args:
            preset (Optional[Union[List[str], Dict[str, str]]]): Preset containing substances or elements of interest.

        Returns:
            self: An instance of self.
        """
        super().__init__()

        if preset is not None:
            self._preset = list(map(lambda x: x.capitalize(), preset))
        else:
            self._preset = ["Carbon", "Hydrogen", "Nitrogen", "Oxygen"]

        self.prefix = ""
        self.suffix = "_mass"
        self.label = [self.prefix + element.lower() + self.suffix for element in self.preset]

    @property
    def preset(self) -> Optional[Union[List[str], Dict[str, str]]]:
        """Get molecular preset. Getter method."""
        return self._preset

    @preset.setter
    def preset(self, new_preset: Optional[Union[List[str], Dict[str, str]]]) -> None:
        """Set molecular preset. Setter method.

        Args:
            new_preset (Optional[Union[List[str], Dict[str, str]]]): List of chemical elements of interest.

        Returns:
            None
        """
        self._preset = new_preset
        if new_preset is not None:
            self.label = [self.prefix + element.lower() + self.suffix for element in self.preset]
        return

    def fit(
        self,
        molecules: Union[
            Union[SMILESMolecule, InChIMolecule, SELFIESMolecule],
            Sequence[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]],
        ],
    ):
        """Generate preset by exploration of molecule sequence. Updates instance state.

        Args:
            molecules (Union[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule],
                        Sequence[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]]]):
                Sequence of molecular instances.

        Returns:
            self (ElementMassFeaturizer): Instance of self with updated state.
        """
        if isinstance(molecules, list):
            unique_elements = set()
            for molecule in molecules:
                unique_elements.update(set(self._get_unique_elements(molecule)))
        else:
            unique_elements = set(self._get_unique_elements(molecules))

        unique_elements = list(unique_elements)
        self.preset = unique_elements

        return self

    def _get_element_mass(
        self, element: str, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> float:
        """
        Get the total mass component of an element in a molecule.

        Args:
            element (str): String representing element name or symbol.
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            element_mass (float): Total mass accounted for by `element`` in `molecule`.
        """
        if len(element) > 2:
            element_mass = [
                self.periodic_table.GetAtomicWeight(atom.GetAtomicNum())
                for atom in molecule.get_atoms(True)
                if self.periodic_table.GetElementName(atom.GetAtomicNum()) == element
            ]
        else:
            element_mass = [
                self.periodic_table.GetAtomicWeight(atom.GetAtomicNum())
                for atom in molecule.get_atoms(True)
                if self.periodic_table.GetElementSymbol(atom.GetAtomicNum()) == element
            ]
        return sum(element_mass)

    def _get_profile(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> List[float]:
        """Generate molecular profile based of preset attribute.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation instance.

        Returns:
            element_masses (List[float]): List of elemental masses.
        """
        element_masses = [
            self._get_element_mass(element=element, molecule=molecule) for element in self.preset
        ]

        return element_masses

    def _get_unique_elements(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> List[str]:
        """
        Get unique elements that make up a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            unique_elements (List[str]): Unique list of element_names or element_symbols in `molecule`.
        """
        unique_elements = [
            self.periodic_table.GetElementName(atom.GetAtomicNum()).capitalize()
            for atom in set(molecule.get_atoms(True))
        ]
        return unique_elements

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance. Get the total mass component for elements in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: Molecular contribution by mass for elements in molecule.
        """
        return np.array(self._get_profile(molecule=molecule)).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ElementMassProportionFeaturizer(ElementMassFeaturizer):
    """Obtain mass proportion for elements in a molecule."""

    def __init__(self, preset: Optional[List[str]] = None):
        """Initialize instance."""
        super().__init__(preset=preset)
        self.prefix = ""
        self.suffix = "_mass_ratio"
        self.label = [self.prefix + element.lower() + self.suffix for element in self.preset]

    def _get_profile(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> List[float]:
        """Generate molecular profile based of preset attribute.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation instance.

        Returns:
            element_proportions (List[float]): List of elemental mass proportions.
        """
        molar_mass = sum(
            [
                self.periodic_table.GetAtomicWeight(atom.GetAtomicNum())
                for atom in molecule.get_atoms()
            ]
        )

        element_proportions = [
            self._get_element_mass(element=element, molecule=molecule) / molar_mass
            for element in self.preset
        ]

        return element_proportions

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance. Get the total mass proportion for elements in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: Molecular proportional contribution by mass for elements in molecule.
        """
        return np.array(self._get_profile(molecule=molecule)).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ElementCountFeaturizer(ElementMassFeaturizer):
    """Get the total mass component of an element in a molecule."""

    def __init__(self, preset: Optional[List[str]] = None):
        """Initialize class."""
        super().__init__(preset=preset)
        self.prefix = "num_"
        self.suffix = "_atoms"

        self.label = [self.prefix + element.lower() + self.suffix for element in self.preset]

    def _get_atom_count(
        self, element: str, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> int:
        """
        Get number of atoms of element in a molecule.

        Args:
            element (str): String representation of a chemical element.
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation instance.

        Returns:
            atom_count (int): Number of atoms of element in molecule.
        """
        atom_count = len(
            [
                atom
                for atom in molecule.get_atoms()
                if (
                    self.periodic_table.GetElementName(atom.GetAtomicNum()) == element
                    or self.periodic_table.GetElementSymbol(atom.GetAtomicNum()) == element
                )
            ]
        )
        return atom_count

    def _get_profile(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> List[int]:
        """Generate number of atoms per element based of preset attribute.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation instance.

        Returns:
            atom_counts (List[int]): List of elemental atom counts.
        """
        atom_counts = [
            self._get_atom_count(element=element, molecule=molecule) for element in self.preset
        ]

        return atom_counts

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance. Get the atom count for elements in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: Molecular contribution by atom count for elements in molecule.
        """
        return np.array([self._get_profile(molecule=molecule)]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ElementCountProportionFeaturizer(ElementCountFeaturizer):
    """Get the proportion of an element in a molecule by atomic count."""

    def __init__(self, preset: Optional[List[str]] = None):
        """Initialize instance."""
        super().__init__(preset=preset)
        self.prefix = ""
        self.suffix = "_atom_ratio"
        self.label = [self.prefix + element.lower() + self.suffix for element in self.preset]

    def _get_profile(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> List[float]:
        """Generate molecular profile based of preset attribute.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation instance.

        Returns:
            element_proportions (List[float]): List of elemental atom proportions.
        """
        num_atoms = molecule.reveal_hydrogens().GetNumAtoms()
        element_proportions = [
            self._get_atom_count(element=element, molecule=molecule) / num_atoms
            for element in self.preset
        ]

        return element_proportions

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize single molecule instance. Get the atom count proportion for elements in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecular representation.

        Returns:
            np.array: Molecular proportional contribution by atom count for elements in molecule.
        """
        return np.array(self._get_profile(molecule=molecule)).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MultipleFeaturizer(AbstractFeaturizer):
    """A featurizer to combine featurizers."""

    def __init__(self, featurizer_list: List[AbstractFeaturizer]):
        """Initialize class instance."""
        super().__init__()
        self.featurizers = featurizer_list

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize a molecule instance via multiple lower-level featurizers.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

        Returns:
            features (np.array), array shape [1, num_featurizers]: Array containing features
                extracted from molecule.
                `num_featurizers` is the number of featurizers passed to MultipleFeaturizer.
        """
        features = [featurizer.featurize(molecule=molecule) for featurizer in self.featurizers]

        return np.concatenate(features, axis=-1)

    def feature_labels(self) -> List[str]:
        """Return feature labels.

        Args:
            None

        Returns:
            List[str]: List of labels for all features extracted by all featurizers.
        """
        labels = list()
        for featurizer in self.featurizers:
            labels += featurizer.feature_labels()

        return labels

    def fit_on_featurizers(self, featurizer_list: List[AbstractFeaturizer]):
        """Fit MultipleFeaturizer instance on lower-level featurizers.

        Args:
            featurizer_list (List[AbstractFeaturizer]): List of lower-level featurizers.

        Returns:
            self : Instance of self with state updated.
        """
        self.featurizers = featurizer_list
        self.label = self.feature_labels()

        return self

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class SMARTSFeaturizer(AbstractFeaturizer):
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
            names: Optional[Union[str, List[str]]]: Preset name(s) of the substructures encoded by the SMARTS strings.
                Predefined presets can be specified as strings, and can be one of:
                    - `heterocyclic`,
                    - `rings`,
                    - `amino`,
                    - `scaffolds`,
                    - `warheads` or
                    - `organic`.
                Defaults to `rings`.
            smarts: Optional[List[str]]: SMARTS strings that are matched with the molecules. Defaults to None.

        Returns:
            self: Instance of self.
        """
        super().__init__()

        if isinstance(names, str):
            try:
                names, smarts = SMARTSPreset(names).preset
            except KeyError:
                raise KeyError(
                    f"`{names}` preset not defined. \
                    Use `heterocyclic`, `rings`, 'amino`, `scaffolds`, `warheads`, or `organic`"
                )
        else:
            assert bool(names) == bool(
                smarts
            ), "Both `names` and `smarts` must either be or not be provided."
            assert len(names) == len(
                smarts
            ), "Both `names` and `smarts` must be lists of the same length."

        self.names = names
        self.smarts = smarts
        self.count = count

        self.prefix = ""
        self.suffix = "_count" if count else "_presence"
        self.label = [self.prefix + element.lower() + self.suffix for element in self.names]

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
        self, new_preset: Optional[Union[str, Dict[str, List[str]], List[List[str]]]]
    ) -> None:
        """Set molecular preset. Setter method.

        Args:
            new_preset (Optional[Union[str, Dict[str, List[str]], List[List[str], List[str]]]]): New preset of interest.
                Could be a:
                    (str) string representing new predefined preset.
                    (Dict[str, List[str]]) map of substance names and SMARTS strings.
                    (List[List[str]]): A list of two lists:
                        First, a list of substance names.
                        Second, a list of corresponding SMARTS strings.

        Returns:
            None
        """

        if new_preset is not None:
            if isinstance(new_preset, str):
                names, smarts = SMARTSPreset(preset=new_preset).preset()
            elif isinstance(new_preset, tuple) or isinstance(new_preset, list):
                names = new_preset[0]
                smarts = new_preset[1]
            else:
                names = new_preset["names"]
                smarts = new_preset["smarts"]

            self.names = names
            self.smarts = smarts

            self.label = [self.prefix + element.lower() + self.suffix for element in self.names]
        else:
            self.names = None
            self.smarts = None
            self.label = [None]
        return

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Return integers representing the frequency or presence of molecular patterns in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

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
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
