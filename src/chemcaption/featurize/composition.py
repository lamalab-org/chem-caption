# -*- coding: utf-8 -*-

"""Featurizers describing the composition of a molecule."""

from collections import Counter
from typing import Dict, List, Optional, Union

import numpy as np
from rdkit.Chem import Descriptors

from chemcaption.featurize.base import PERIODIC_TABLE, AbstractFeaturizer
from chemcaption.featurize.utils import join_list_elements
from chemcaption.molecules import Molecule

# Implemented composition-related featurizers

__all__ = [
    "MolecularFormulaFeaturizer",
    "MolecularMassFeaturizer",
    "MonoisotopicMolecularMassFeaturizer",
    "ElementMassFeaturizer",
    "ElementMassProportionFeaturizer",
    "ElementCountFeaturizer",
    "ElementCountProportionFeaturizer",
    "AtomCountFeaturizer",
    "DegreeOfUnsaturationFeaturizer",
]


class MolecularFormulaFeaturizer(AbstractFeaturizer):
    """Get the molecular formula of a molecule."""

    def __init__(self):
        """Initialize class."""
        super().__init__()

        self._names = [
            {
                "noun": "molecular formula",
            }
        ]

    def feature_labels(self) -> List[str]:
        return ["molecular_formula"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the molecular formular of a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (str): Molecular formular of `molecule`.
        """
        return np.array([molecule.get_composition()]).reshape((1, 1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MolecularMassFeaturizer(AbstractFeaturizer):
    """Get the molecular mass of a molecule."""

    def __init__(self):
        """Get the molecular mass of a molecule."""
        super().__init__()

        self.template = (
            "What is the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "molecular mass",
            }
        ]

    def feature_labels(self) -> List[str]:
        return ["molecular_mass"]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Get the molecular mass of a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            molar_mass (float): Molecular mass of `molecule`.
        """
        molar_mass = Descriptors.MolWt(molecule.rdkit_mol)
        return np.array([molar_mass]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MonoisotopicMolecularMassFeaturizer(AbstractFeaturizer):
    """Get the monoisotopic molecular mass of a molecule."""

    def __init__(self):
        """Instantiate instance."""
        super().__init__()

        self.template = (
            "What is the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "monoisotopic molecular mass",
            }
        ]

    def feature_labels(self) -> List[str]:
        return ["monoisotopic_molecular_mass"]

    def featurize(
        self,
        molecule: Molecule,
    ) -> np.array:
        """
        Featurize single molecule instance. Get the monoisotopic molecular mass of a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            molar_mass (float): Monoisotopic molecular mass of `molecule`.
        """
        monoisotopic_molar_mass = Descriptors.ExactMolWt(molecule.rdkit_mol)
        return np.array([monoisotopic_molar_mass]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

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
        """
        super().__init__()

        if preset is not None:
            self._preset = list(map(lambda x: x.capitalize(), preset))
        else:
            self._preset = ["Carbon", "Hydrogen", "Nitrogen", "Oxygen"]

        self.template = (
            "What is the {PROPERTY_NAME} for the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )

    def get_names(self) -> List[Dict[str, str]]:
        return [{"noun": "total mass of " + join_list_elements(self.preset)}]

    def feature_labels(self) -> List[str]:
        return [element.lower() + "_mass" for element in self.preset]

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
            None.
        """
        self._preset = new_preset
        return

    def fit(
        self,
        molecules: Union[
            Molecule,
            List[Molecule],
        ],
    ):
        """Generate preset by exploration of molecule sequence. Updates instance state.

        Args:
            molecules (Union[Molecule, List[Molecule]]): Sequence of molecular instances.

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

    def _get_element_mass(self, element: str, molecule: Molecule) -> float:
        """
        Get the total mass component of an element in a molecule.

        Args:
            element (str): String representing element name or symbol.
            molecule (Molecule): Molecular representation.

        Returns:
            element_mass (float): Total mass accounted for by `element`` in `molecule`.
        """
        if len(element) > 2:
            element_mass = [
                PERIODIC_TABLE.GetAtomicWeight(atom.GetAtomicNum())
                for atom in molecule.get_atoms(True)
                if PERIODIC_TABLE.GetElementName(atom.GetAtomicNum()) == element
            ]
        else:
            element_mass = [
                PERIODIC_TABLE.GetAtomicWeight(atom.GetAtomicNum())
                for atom in molecule.get_atoms(True)
                if PERIODIC_TABLE.GetElementSymbol(atom.GetAtomicNum()) == element
            ]
        return sum(element_mass)

    def _get_profile(self, molecule: Molecule) -> List[float]:
        """Generate molecular profile based of preset attribute.

        Args:
            molecule (Molecule): Molecular representation instance.

        Returns:
            element_masses (List[float]): List of elemental masses.
        """
        element_masses = [
            self._get_element_mass(element=element, molecule=molecule) for element in self.preset
        ]

        return element_masses

    def _get_unique_elements(self, molecule: Molecule) -> List[str]:
        """
        Get unique elements that make up a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            unique_elements (List[str]): Unique list of element_names or element_symbols in `molecule`.
        """
        unique_elements = [
            PERIODIC_TABLE.GetElementName(atom.GetAtomicNum()).capitalize()
            for atom in set(molecule.get_atoms(True))
        ]
        return unique_elements

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the total mass component for elements in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: Molecular contribution by mass for elements in molecule.
        """
        return np.array(self._get_profile(molecule=molecule)).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

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

    def get_names(self) -> List[Dict[str, str]]:
        return [{"noun": "mass proportion of " + join_list_elements(self.preset)}]

    def feature_labels(self) -> List[str]:
        """
        Return list of feature labels.

        Args:
            None.

        Returns:
            List[str]: List of feature labels.
        """
        return [self.prefix + element.lower() + self.suffix for element in self.preset]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the total mass proportion for elements in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: Molecular proportional contribution by mass for elements in molecule.
        """
        molar_mass = Descriptors.MolWt(molecule.rdkit_mol)
        return np.array(self._get_profile(molecule=molecule)).reshape((1, -1)) / molar_mass

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ElementCountFeaturizer(ElementMassFeaturizer):
    """Get the total mass component of an element in a molecule."""

    def __init__(self, preset: Optional[List[str]] = None):
        """Initialize class."""
        super().__init__(preset=preset)

    def feature_labels(self) -> List[str]:
        return ["num_" + element.lower() + "_atoms" for element in self.preset]

    def get_names(self):
        return [{"noun": "atom count of " + join_list_elements(self.preset)}]

    def _get_atom_count(self, element: str, molecule: Molecule) -> int:
        """
        Get number of atoms of element in a molecule.

        Args:
            element (str): String representation of a chemical element.
            molecule (Molecule): Molecular representation instance.

        Returns:
            atom_count (int): Number of atoms of element in molecule.
        """
        atom_count = len(
            [
                atom
                for atom in molecule.get_atoms()
                if (
                    PERIODIC_TABLE.GetElementName(atom.GetAtomicNum()) == element
                    or PERIODIC_TABLE.GetElementSymbol(atom.GetAtomicNum()) == element
                )
            ]
        )
        return atom_count

    def _get_profile(self, molecule: Molecule) -> List[int]:
        """Generate number of atoms per element based of preset attribute.

        Args:
            molecule (Molecule): Molecular representation instance.

        Returns:
            atom_counts (List[int]): List of elemental atom counts.
        """
        atom_counts = [
            self._get_atom_count(element=element, molecule=molecule) for element in self.preset
        ]

        return atom_counts

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the atom count for elements in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: Molecular contribution by atom count for elements in molecule.
        """
        return np.array([self._get_profile(molecule=molecule)]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class ElementCountProportionFeaturizer(ElementCountFeaturizer):
    """Get the proportion of an element in a molecule by atomic count."""

    def __init__(self, preset: Optional[List[str]] = None):
        """Initialize instance.

        Args:
            preset (Optional[List[str]]): None or List of strings. Containing the names of elements of interest.
                Defaults to `None`.
        """
        super().__init__(preset=preset)

    def get_names(self):
        return [{"noun": "relative atom count of " + join_list_elements(self.preset)}]

    def feature_labels(self) -> List[str]:
        return [element.lower() + "_atom_ratio" for element in self.preset]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the atom count proportion for elements in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: Molecular proportional contribution by atom count for elements in molecule.
        """
        num_atoms = len(molecule.get_atoms(hydrogen=True))
        return np.array(self._get_profile(molecule=molecule)).reshape((1, -1)) / num_atoms

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class AtomCountFeaturizer(ElementCountFeaturizer):
    """Get the number of atoms in a molecule."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()
        self._names = [
            {
                "noun": "total number of atoms",
            }
        ]

    def get_names(self):
        return [{"noun": "total number of atoms"}]

    def feature_labels(self) -> List[str]:
        return ["num_atoms"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the atom count proportion for elements in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: Number of atoms in `molecule`.
        """
        return np.array([len(molecule.get_atoms(hydrogen=True))]).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class DegreeOfUnsaturationFeaturizer(AbstractFeaturizer):
    """Return the degree of unsaturation."""

    def __init__(self):
        """Instantiate class.

        Args:
            None.
        """
        super().__init__()

        self._names = [
            {
                "noun": "degree of unsaturation",
            }
        ]

    def feature_labels(self) -> List[str]:
        return ["degree_of_unsaturation"]

    def _get_degree_of_unsaturation_for_mol(self, molecule: Molecule):
        """Return the degree of unsaturation for a molecule.

        .. math::
            {\displaystyle \mathrm {DU} =1+{\tfrac {1}{2}}\sum n_{i}(v_{i}-2)}

        where ni is the number of atoms with valence vi.

        Args:
            molecule (Molecule): Molecule instance.

        Returns:
            (int): Degree of unsaturation.
        """
        # add hydrogens
        mol = molecule.reveal_hydrogens()
        valence_counter = Counter()
        for atom in mol.GetAtoms():
            valence_counter[atom.GetExplicitValence()] += 1
        du = 1 + 0.5 * sum([n * (v - 2) for v, n in valence_counter.items()])
        return du

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Returns the degree of unsaturation of a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: degree of unsaturation.
        """
        return np.array([self._get_degree_of_unsaturation_for_mol(molecule)]).reshape((1, 1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Kevin Maik Jablonka"]
