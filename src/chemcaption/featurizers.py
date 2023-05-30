# -*- coding: utf-8 -*-

"""Utility imports."""

from abc import ABC, abstractmethod
from typing import List, Sequence, Union

import numpy as np
import rdkit
from rdkit.Chem import Lipinski, rdMolDescriptors

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


if __name__ == "__main__":
    from selfies import encoder
    inchi = "InChI=1S/C6H5NO2/c8-7(9)6-4-2-1-3-5-6/h1-5H"
    smiles = "CN(C)[C@H]1[C@@H]2C[C@H]3C(=C(O)c4c(O)cccc4[C@@]3(C)O)C(=O)[C@]2(O)C(=O)\C(=C(/O)NCN5CCCC5)C1=O"
    smiles_2 = 'OC(=O)CCC(O)=O.FC(F)(F)c1ccc2Sc3ccccc3N(CCCN4CCN(CCC5OCCCO5)CC4)c2c1'
    selfies_form = encoder(smiles)
    repr_type = "inchi"

    inchi_mol = InChIMolecule(inchi)
    smiles_mol = SMILESMolecule(smiles)
    smiles_mol_2 = SMILESMolecule(smiles_2)

    mol_list = [inchi_mol, smiles_mol, smiles_mol_2]

    preset = ["Chromium", "Phosphorus", 'Carbon', "Hydrogen", "F"]

    bond = ElementCountProportionFeaturizer()
    bond.fit(mol_list)

    #print(bond.featurize(smiles_mol))
    print(bond.batch_featurize(mol_list))
    print(bond.feature_labels())