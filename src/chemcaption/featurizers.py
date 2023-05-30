# -*- coding: utf-8 -*-

"""Utility imports."""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Sequence, Union

import numpy as np
import rdkit
from rdkit.Chem import Lipinski, rdMolDescriptors
from selfies import encoder

from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

"""Abstract classes."""


class AbstractFeaturizer(ABC):
    """Base class for lower level Featurizers.

    Args:
        None

    Returns:
        None
    """

    def __init__(self):
        """Initialize periodic table."""
        self.periodic_table = rdkit.Chem.GetPeriodicTable()
        self._label = list()

    @abstractmethod
    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """Featurize single Molecule instance."""
        raise NotImplementedError

    def batch_featurize(self, molecules) -> np.array:
        """
        Featurize a sequence of Molecule objects.

        Args:
            molecules (Sequence[MoleculeType]): A sequence of molecule representations.

        Returns:
            np.array: An array of features for each molecule instance.
        """

        return np.concatenate([self.featurize(molecule) for molecule in molecules])

    def text_featurize(self, molecule):
        """Embed features in Prompt instance."""
        return None

    def batch_text_featurize(
        self, molecules: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ):
        """Embed features in Prompt instance for multiple molecules."""
        return None

    @abstractmethod
    def implementors(self) -> List[str]:
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        raise NotImplementedError

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, new_label):
        self._label = new_label
        return

    def feature_labels(self) -> List[str]:
        return self.label

    def citations(self):
        return None


"""
Lower level featurizer classes.

1. NumRotableBondsFeaturizer
2. BondRotabilityFeaturizer
3. HAcceptorCountFeaturizer
4. HDonorCountFeaturizer
5. ElementMassFeaturizer
6. ElementCountFeaturizer
7. ElementMassProportionFeaturizer
8. ElementCountProportionFeaturizer
"""


class NumRotableBondsFeaturizer(AbstractFeaturizer):
    def __init__(self):
        """Count the number of rotable (single, non-terminal) bonds in a molecule."""
        super().__init__()
        self.label = ["num_rotable_bonds"]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Count the number of rotable (single, non-terminal) bonds in a molecule.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            num_rotable (np.array): Number of rotable bonds in molecule.
        """
        num_rotable = rdMolDescriptors.CalcNumRotatableBonds(molecule.rdkit_mol, strict=True)
        return np.array([num_rotable])

    def implementors(self):
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class BondRotabilityFeaturizer(AbstractFeaturizer):
    def __init__(self):
        super().__init__()
        self.label = ["rotable_proportion", "non_rotable_proportion"]

    def _get_bond_types(self, molecule):
        num_bonds = len(molecule.rdkit_mol.GetBonds())
        num_rotable = rdMolDescriptors.CalcNumRotatableBonds(molecule.rdkit_mol, strict=True)
        num_non_rotable = num_bonds - num_rotable

        bond_distribution = [num_rotable/num_bonds, num_non_rotable/num_bonds]

        return bond_distribution

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        return np.reshape(np.array(self._get_bond_types(molecule=molecule)), newshape=(1, -1))

    def implementors(self):
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class HAcceptorCountFeaturizer(AbstractFeaturizer):
    def __init__(self):
        """Get the number of Hydrogen bond acceptors present in a molecule."""
        super().__init__()
        self.label = ["num_hydrogen_bond_acceptors"]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Get the number of Hydrogen bond acceptors present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (np.array): Number of Hydrogen bond acceptors present in `molecule`.
        """
        return np.array([Lipinski.NumHAcceptors(molecule.rdkit_mol)])

    def implementors(self) -> List[str]:
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class HDonorCountFeaturizer(AbstractFeaturizer):
    def __init__(self):
        """Get the number of Hydrogen bond donors present in a molecule."""
        super().__init__()
        self.label = ["num_hydrogen_bond_donors"]

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Get the number of Hydrogen bond donors present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            np.array: Number of Hydrogen bond donors present in `molecule`.
        """
        return np.array([Lipinski.NumHDonors(molecule.rdkit_mol)])

    def implementors(self) -> List[str]:
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]

    def citations(self):
        return None


class MolecularMassFeaturizer(AbstractFeaturizer):
    def __init__(self):
        """Get the molecular mass of a molecule."""
        super().__init__()
        self.label = ["molecular_mass"]

    def featurize(
        self,
        molecule,
    ) -> np.array:
        """
        Get the molecular mass of a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            molar_mass (float): Molecular mass of `molecule`.
        """
        molar_mass = sum(
            [
                self.periodic_table.GetAtomicWeight(atom.GetAtomicNum())
                for atom in molecule.get_atoms()
            ]
        )
        return np.array([molar_mass])

    def implementors(self) -> List[str]:
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class ElementMassFeaturizer(AbstractFeaturizer):
    def __init__(self, preset=None):
        """Get the total mass component of an element in a molecule."""
        super().__init__()

        if preset is not None:
            self._preset = list(map(lambda x: x.capitalize(), preset))
        else:
            self._preset = [
                "Carbon", "Hydrogen", "Nitrogen", "Oxygen"
            ]

        self.label = [element.lower() + "_mass" for element in self.preset]

    @property
    def preset(self):
        return self._preset

    @preset.setter
    def preset(self, new_preset):
        self._preset = new_preset
        self.label = [element.lower() + "_mass" for element in self.preset]
        return

    def fit(self, molecules):
        if isinstance(molecules, list):
            unique_elements = set()
            for molecule in molecules:
                unique_elements.update(set(self._get_unique_elements(molecule)))
        else:
            unique_elements = set(self._get_unique_elements(molecules))

        unique_elements = list(unique_elements)
        self.preset = unique_elements

        return self

    def _get_element_mass(self, element, molecule):
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

    def _get_profile(self, molecule):
        element_masses = list()
        for element in self.preset:
            element_mass = self._get_element_mass(element=element, molecule=molecule)
            element_masses.append(element_mass)

        return element_masses

    def _get_unique_elements(self, molecule=None):
        """
        Get unique elements that make up a molecule.

        Args:
            atomic_info (List[namedtuple]): List of ElementalInformation namedtuples.
            molecule (Molecule): Molecular representation.

        Returns:
            unique_elements (List[str]): Unique list of element_names or element_symbols in `molecule`.
        """

        unique_elements = [
            self.periodic_table.GetElementName(atom.GetAtomicNum()).capitalize() for atom in set(molecule.get_atoms(True))
        ]
        return unique_elements

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        return np.reshape(np.array(self._get_profile(molecule=molecule)), newshape=(1, -1))

    def implementors(self) -> List[str]:
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class ElementMassProportionFeaturizer(ElementMassFeaturizer):
    def __init__(self, preset=None):
        super().__init__(preset=preset)
        print(self.preset)
        self.label = [element.lower() + "_mass_ratio" for element in self.preset]

    def _get_profile(self, molecule):
        molar_mass = sum(
            [
                self.periodic_table.GetAtomicWeight(atom.GetAtomicNum())
                for atom in molecule.get_atoms()
            ]
        )
        element_proportions = list()
        element_labels = list()

        for element in self.preset:
            element_proportion = self._get_element_mass(element=element, molecule=molecule) / molar_mass
            element_proportions.append(element_proportion)
            element_labels.append(element.lower() + "_mass_ratio")

        self.label = element_labels

        return element_proportions

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        return np.reshape(np.array(self._get_profile(molecule=molecule)), newshape=(1, -1))

    def implementors(self) -> List[str]:
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class ElementCountFeaturizer(ElementMassFeaturizer):
    def __init__(self, preset=None):
        """Get the total mass component of an element in a molecule."""
        super().__init__(preset=preset)

        self.label = ["num_" + element.lower() + "_atoms" for element in self.preset]

    def _get_atom_count(self, element, molecule):
        atom_count = len(
            [
                atom for atom in molecule.get_atoms() if
                (
                        self.periodic_table.GetElementName(atom.GetAtomicNum()) == element or
                        self.periodic_table.GetElementSymbol(atom.GetAtomicNum()) == element
                )
            ]
        )
        return atom_count

    def _get_profile(self, molecule):
        atom_counts = list()
        for element in self.preset:
            atom_count = self._get_atom_count(element=element, molecule=molecule)
            atom_counts.append(atom_count)

        return atom_counts

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        return np.array([self._get_profile(molecule=molecule)])

    def implementors(self) -> List[str]:
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]


class ElementCountProportionFeaturizer(ElementCountFeaturizer):
    def __init__(self, preset=None):
        super().__init__(preset=preset)
        print(self.preset)
        self.label = [element.lower() + "_atom_ratio" for element in self.preset]

    def _get_profile(self, molecule):
        molar_mass = sum(
            [
                self.periodic_table.GetAtomicWeight(atom.GetAtomicNum())
                for atom in molecule.get_atoms()
            ]
        )
        element_proportions = list()
        element_labels = list()

        for element in self.preset:
            element_proportion = self._get_atom_count(element=element, molecule=molecule) / molar_mass
            element_proportions.append(element_proportion)
            element_labels.append(element.lower() + "_atom_ratio")

        self.label = element_labels

        return element_proportions

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        return np.reshape(np.array(self._get_profile(molecule=molecule)), newshape=(1, -1))

    def implementors(self) -> List[str]:
        """
        Return functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]





















class BondFeaturizer(AbstractFeaturizer):
    """Lower level Featurizer for bond information."""

    def __init__(self):
        """Initialize class."""
        super(BondFeaturizer, self).__init__()

    def count_bonds(self, molecule, bond_type="ALL"):
        """
        Count the frequency of a bond_type in a molecule.

        Args:
            molecule (Molecule): Molecule representation.
            bond_type (str): Type of bond to enumerate. If `all`, enumerates all bonds irrespective of type.
                Default (ALL).

        Returns:
            num_bonds (int): Number of occurrences of `bond_type` in molecule.
        """
        bond_type = bond_type.upper()

        all_bonds = self.get_bonds(molecule)

        if bond_type == "ALL":
            num_bonds = len(all_bonds)
        else:
            num_bonds = len([bond for bond in all_bonds if bond == bond_type])

        return num_bonds

    def count_rotable_bonds(self, molecule):
        """
        Count the number of rotable (single, non-terminal) bonds in a molecule.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            num_rotable (int): Number of rotable bonds in molecule.
        """
        num_rotable = rdMolDescriptors.CalcNumRotatableBonds(molecule.rdkit_mol, strict=True)
        return num_rotable

    def get_bonds(
        self,
        molecule=None,
    ):
        """
        Extract all individual bonds present in a molecule.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            bonds (List[BondType]): List of all bonds present in molecule.
        """
        bonds = [str(bond.GetBondType()).split(".")[-1] for bond in molecule.rdkit_mol.GetBonds()]

        return bonds

    def get_bond_distribution(self, molecule=None, normalize=True):
        """Return a frequency distribution for the bonds present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.
            normalize (bool): Normalize the freequency or not. Default (True).

        Returns:
            bond_distribution (dict[str, int|float]): Map of BondType string representation to BondType frequency.

        """
        all_bonds = self.get_bonds(molecule)
        num_bonds = len(all_bonds)

        if normalize:
            bond_distribution = {bond: all_bonds.count(bond) / num_bonds for bond in all_bonds}
        else:
            bond_distribution = {bond: all_bonds.count(bond) for bond in all_bonds}

        return bond_distribution

    def get_unique_bond_types(self, molecule):
        """
        Get the unique bond types present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            unique_bonds (set): Set of unique bonds present in `molecule`.
        """
        bonds = self.get_bonds(molecule)
        unique_bonds = set([str(bond).split(".")[-1] for bond in bonds])

        return unique_bonds

    def count_hydrogen_acceptors(self, molecule):
        """
        Get the number of Hydrogen bond acceptors present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (int): Number of Hydrogen bond acceptors present in `molecule`.
        """
        return Lipinski.NumHAcceptors(molecule.rdkit_mol)

    def count_hydrogen_donors(self, molecule):
        """
        Get the number of Hydrogen bond donors present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            (int): Number of Hydrogen bond donors present in `molecule`.
        """
        return Lipinski.NumHDonors(molecule.rdkit_mol)


class ElementFeaturizer(AbstractFeaturizer):
    """Lower level Featurizer for elemental information."""

    def __init__(self):
        """Initialize class."""
        super(ElementFeaturizer, self).__init__()

    def get_elements_info(self, molecule):
        """
        Get information on all elemental atoms present in a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            atoms_info (List(namedtuple)): List of ElementalInformation namedtuple instances containing:
                element_name (str): Name of element atom is made of.
                element_symbol (str): Chemical symbol for the element.
                atomic_number (int): Number of protons in atom nucelus.
                atomic_mass (float): Number of protons + Number of neutrons
        """
        molecule.reveal_hydrogens()
        atoms_info = namedtuple(
            "ElementalInformation",
            ["element_name", "element_symbol", "atomic_number", "atomic_mass"],
        )

        atoms_info = [
            atoms_info(
                self.periodic_table.GetElementName(atom.GetAtomicNum()),
                self.periodic_table.GetElementSymbol(atom.GetAtomicNum()),
                atom.GetAtomicNum(),
                self.periodic_table.GetAtomicWeight(atom.GetAtomicNum()),
            )
            for atom in molecule.get_atoms()
        ]

        return atoms_info

    def get_unique_elements(self, atomic_info=None, molecule=None):
        """
        Get unique elements that make up a molecule.

        Args:
            atomic_info (List[namedtuple]): List of ElementalInformation namedtuples.
            molecule (Molecule): Molecular representation.

        Returns:
            unique_elements (List[str]): Unique list of element_names or element_symbols in `molecule`.
        """
        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule)

        unique_elements = [inner_tuple.element_name for inner_tuple in set(atomic_info)]
        return unique_elements

    def get_element_frequency(self, element, molecule=None, atomic_info=None):
        """
        Get the number of times atoms of an element occur in a molecule.

        Args:
            element (str): Element name or symbol.
            molecule (Molecule): Molecular representation.
            atomic_info (List[namedtuple]): List of ElementalInformation instances containing info
                on all atomic contents of molecule.

        Returns:
            element_count (int): Number of occurrences of atoms of `element` in `molecule`.
        """
        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule=molecule)

        element, element_index = element.capitalize(), (0 if len(element) > 2 else 1)

        element_count = len(
            [
                element_info[element_index]
                for element_info in atomic_info
                if element_info[element_index] == element.capitalize()
            ]
        )
        return element_count


class MassFeaturizer(ElementFeaturizer):
    """Lower level Featurizer for mass-related information."""

    def __init__(self):
        """Initialize class."""
        super(MassFeaturizer, self).__init__()

    def get_molar_mass(
        self,
        atomic_info=None,
        molecule=None,
    ):
        """
        Get the molar mass of a molecule.

        Args:
            atomic_info (List[namedtuple]):  List of ElementalInformation instances containing info on
                all atomic contents of molecule.
            molecule (Molecule): Molecular representation.

        Returns:
            molar_mass (float): Molecular mass of `molecule`.
        """
        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule)
        molar_mass = sum([inner_tuple.atomic_mass for inner_tuple in atomic_info])
        return molar_mass

    def get_total_element_mass(self, element=None, molecule=None, atomic_info=None):
        """
        Get the total mass component of an element in a molecule.

        Args:
            element (str): String representing name or symbol of element.
            molecule (Molecule): Molecular representation.
            atomic_info (List[namedtuple]):  List of ElementalInformation instances containing info on
                all atomic contents of molecule.

        Returns:
            element_mass (float): Total mass accounted for by `element` in `molecule`.
        """
        element = element.capitalize()
        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule=molecule)
        element_mass = sum(
            [
                inner_tuple.atomic_mass
                for inner_tuple in atomic_info
                if (inner_tuple.element_name == element or inner_tuple.element_symbol == element)
            ]
        )
        return element_mass

