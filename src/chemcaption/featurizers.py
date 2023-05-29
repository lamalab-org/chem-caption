# -*- coding: utf-8 -*-

"""Utility imports."""

from abc import ABC, abstractmethod
from collections import namedtuple

import rdkit
from rdkit.Chem import Lipinski, rdMolDescriptors

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

    @abstractmethod
    def featurize(self, molecule):
        """Featurize single Molecule instance."""
        raise NotImplementedError

    @abstractmethod
    def text_featurize(self, molecule):
        """Embed features in Prompt instance."""
        raise NotImplementedError

    @abstractmethod
    def batch_featurize(self, molecules):
        """Featurize both collection of Molecule objects."""
        raise NotImplementedError

    @abstractmethod
    def batch_text_featurize(self, molecules):
        """Embed features in Prompt instance for multiple molecules."""
        raise NotImplementedError


"""
Lower level featurizer classes.

1. BondFeaturizer
2. ElementFeaturizer
3. MassFeaturizer
"""


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


"""Mega featurizer class."""


class MoleculeFeaturizer(BondFeaturizer, MassFeaturizer, ElementFeaturizer):
    """Molecule-centered Featurizer class."""

    def __init__(self):
        """Initialize class."""
        super(MoleculeFeaturizer, self).__init__()

    def stream_featurize(self, molecule=None, atomic_info=None):
        """
        Generate feature vector for Molecule object.

        Args:
            molecule (Molecule): Molecule representation.
            atomic_info (namedtuple): ElementalInformation instance containing info on all atomic contents of molecule.

        Returns:
            features [namedtuple]: NamedTuple containing:
                molecular_mass (float): Molar mass of molecule.
                element_count (int): Number of atoms of an element present in molecule.
                element_mass (float): Atomic mass of element * `element_count`.
                element_count_proportion (float): Percentage of total number of atoms made up by element.
                element_mass_proportion (float): Percentage of molecular mass made up by element.
                num_<BondType>_bond (int): Number of occurrences of each bond type.
                <BondType>_bond_proportion (float): Fraction/proportion of occurrences of each bond type.
                num_rotable_bonds (int): Number of rotable bonds in molecule.
                rotable_bond_proportion (float): Percentage of rotable bonds in molecule.
                num_non_rotale_bonds (int): Number of non-rotable bond.
                non_rotable_bond_proportion (float): Percentage of non-rotable bonds in molecule.
                num_hydrogen_donors (int): Number of Hydrogen bond donors in molecule.
                num_hydrogen_acceptors (int): Number of Hydrogen bond acceptors in molecule.

        """
        feature_names = [
            "molecular_mass",
        ]
        feature_values = list()

        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule=molecule)

        molecular_mass = self.get_molar_mass(atomic_info=atomic_info)
        feature_values.append(molecular_mass)

        element_features, element_values = self._get_elemental_profile(
            molecule=molecule, atomic_info=atomic_info
        )

        feature_names += element_features
        feature_values += element_values

        features = namedtuple("MolecularInformation", feature_names)

        return features(*feature_values)

    def _get_elemental_profile(self, molecule=None, atomic_info=None, normalize=True):
        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule=molecule)
        unique_elements = self.get_unique_elements(atomic_info=atomic_info)

        element_features = list()
        element_values = list()
        molar_mass = self.get_molar_mass(atomic_info=atomic_info)

        for element in unique_elements:
            element_features.append(f"{element.lower()}_count")
            element_count = self.get_element_frequency(
                element=element, molecule=molecule, atomic_info=atomic_info
            )
            element_values.append(element_count)

            element_features.append(f"{element.lower()}_total_mass")
            element_mass = self.get_total_element_mass(
                element=element, molecule=molecule, atomic_info=atomic_info
            )
            element_values.append(element_mass)

            if normalize:
                element_features.append(f"{element.lower()}_count_proportion")
                element_values.append(element_count / len(atomic_info))

                element_features.append(f"{element.lower()}_mass_proportion")
                element_values.append(
                    self.get_total_element_mass(element=element, atomic_info=atomic_info)
                    / molar_mass
                )

        for norm in [True, False]:
            bond_types, bond_counts = zip(
                *self.get_bond_distribution(molecule, normalize=norm).items()
            )
            element_features += list(
                map(
                    lambda x: f"{x.lower()}_bond_proportion" if norm else f"num_{x.lower()}_bonds",
                    bond_types,
                )
            )
            element_values += bond_counts

        num_rotable = self.count_rotable_bonds(molecule=molecule)
        num_bonds = self.count_bonds(molecule=molecule, bond_type="ALL")
        num_non_rotable = num_bonds - num_rotable

        rotable_proportion = num_rotable / num_bonds
        non_rotable_proportion = num_non_rotable / num_bonds

        element_features.append("num_rotable_bonds")
        element_features.append("rotable_bonds_proportion")

        element_values.append(num_rotable)
        element_values.append(rotable_proportion)

        element_features.append("num_non_rotable_bonds")
        element_features.append("non_rotable_bonds_proportion")

        element_values.append(num_non_rotable)
        element_values.append(non_rotable_proportion)

        element_features.append("num_hydrogen_donors")
        element_features.append("num_hydrogen_acceptors")

        num_hydrogen_donors = self.count_hydrogen_donors(molecule)
        num_hydrogen_acceptors = self.count_hydrogen_acceptors(molecule)

        element_values.append(num_hydrogen_donors)
        element_values.append(num_hydrogen_acceptors)

        return element_features, element_values

    def featurize(self, molecules=None, molecular_info=None):
        """
        Featurize both collection of Molecule objects or single Molecule instance.

        Args:
            molecules (Union[Molecule, List[Molecules]): Single or collection of Molecule instance(s).
            molecular_info (namedtuple): Single or collection of ElementalInformation namedtuple instance(s).

        Returns:
            features (Union[namedtuple, List[namedtuple]): Single or collection of
                MolecularInformation namedtuple instance(s).
        """
        if isinstance(molecules, list):
            features = self.batch_featurize(molecules=molecules, molecular_info_dump=molecular_info)
            features = features[0] if len(features) == 1 else features
        else:
            features = self.stream_featurize(molecule=molecules, atomic_info=molecular_info)

        return features

    def batch_featurize(self, molecules=None, molecular_info_dump=None):
        """
        Featurize a collection of Molecule objects.

        Args:
            molecules (Molecule): Molecular representation.
            molecular_info_dump (List[namedtuple]): Collection of ElementalInformation instances containing
                elemental information for each molecule.

        Returns:
            List[namedtuple]: List of MolecularInformation namedtuple objects for each molecule.
        """
        if molecular_info_dump is None:
            molecular_info_dump = [
                self.get_elements_info(molecule=molecule) for molecule in molecules
            ]
        return [
            self.stream_featurize(molecule=molecule, atomic_info=molecular_info)
            for molecule, molecular_info in zip(molecules, molecular_info_dump)
        ]

    def text_featurize(self, molecule):
        """Embed features in Prompt instance."""
        return None

    def batch_text_featurize(self, molecules):
        """Embed features in Prompt instance for multiple molecules."""
        return None
