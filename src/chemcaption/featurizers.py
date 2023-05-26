# -*- coding: utf-8 -*-

"""Utility imports."""
import random

import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Lipinski

from abc import abstractmethod, ABC
from selfies import encoder, decoder

from collections import namedtuple

from molecules import Molecule


"""Abstract classes."""

class FeaturizerBase(ABC):
    def __init__(self):
        self.periodic_table = rdkit.Chem.GetPeriodicTable()


class AbstractFeaturizer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def featurize(self, molecule):
        raise NotImplementedError

    @abstractmethod
    def text_featurize(self, molecule):
        raise NotImplementedError

    @abstractmethod
    def batch_featurize(self, molecules):
        raise NotImplementedError

    @abstractmethod
    def batch_text_featurize(self, molecules):
        raise NotImplementedError



"""Lower level featurizer classes."""


class BondFeaturizer(FeaturizerBase):
    def __init__(self):
        super(BondFeaturizer, self).__init__()

    def count_bonds(self, molecule, bond_type="SINGLE"):
        bond_type = bond_type.upper()
        num_bonds = sum(
            [
                (True if str(bond.GetBondType()).split(".")[-1] == bond_type else False)
                for bond in molecule.get_rdkit_mol().GetBonds()
            ]
        )
        return num_bonds

    def count_rotable_bonds(self, molecule):
        return rdMolDescriptors.CalcNumRotatableBonds(molecule.rdkit_mol,strict=True)

    def get_bonds(
        self,
        molecule=None,
    ):
        bonds = [
            str(bond.GetBondType()).split(".")[-1] for bond in molecule.rdkit_mol.GetBonds()
        ]

        return bonds

    def get_bond_distribution(self, molecule=None,):
        molecular_info = self.get_bonds(molecule)

        return {bond: molecular_info.count(bond) for bond in molecular_info}

    def get_unique_bond_types(self, molecule):
        bonds = self.get_bonds(molecule)
        unique_bonds = [str(bond).split(".")[-1] for bond in bonds]

        return set(unique_bonds)

    def count_hydrogen_acceptors(self, molecule):
        return  Lipinski.NumHAcceptors(molecule.rdkit_mol)

    def count_hydrogen_donors(self, molecule):
        return Lipinski.NumHDonors(molecule.rdkit_mol)


class ElementFeaturizer(FeaturizerBase):
    def __init__(self):
        super(ElementFeaturizer, self).__init__()
    def get_elements_info(self, molecule):
        molecule.reveal_hydrogens()
        atoms_info = namedtuple("ElementalInformation", ["element", "symbol", "atomic_number", "atomic_mass"])

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
        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule)

        unique_elements = [TUPLE.element for TUPLE in set(atomic_info)]
        return unique_elements

    def get_element_frequency(self, element, molecule=None, atomic_info=None):
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
    def __init__(self):
        super(MassFeaturizer, self).__init__()

    def get_molar_mass(
        self,
        atomic_info=None,
        molecule=None,
    ):
        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule)
        molar_mass = sum([TUPLE.atomic_mass for TUPLE in atomic_info])
        return molar_mass

    def get_total_element_mass(self, element=None, molecule=None, atomic_info=None):
        element = element.capitalize()
        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule=molecule)
        element_mass = sum(
            [
                TUPLE.atomic_mass for TUPLE in atomic_info if (TUPLE.element == element or TUPLE.symbol == element)
            ]
        )
        return element_mass


"""Mega featurizer class."""

class MoleculeFeaturizer(BondFeaturizer, MassFeaturizer, ElementFeaturizer):
    def __init__(self):
        super(MoleculeFeaturizer, self).__init__()

    def stream_featurize(self, molecule=None, atomic_info=None):
        """
        Generates feature vector for Molecule object.

        Parameters
        ----------
        molecule [Molecule]: Molecule object.

        Returns
        -------
        features [namedtuple]: Tuple containing:
            atomic_mass [float]: Sum of number of neutrons and number of protons

        """
        feature_names = [
            "molecular_mass",
        ]
        feature_values = list()

        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule=molecule)

        molecular_mass = self.get_molar_mass(atomic_info=atomic_info)
        feature_values.append(molecular_mass)

        element_features, element_values = self.get_elemental_profile(molecule=molecule, atomic_info=atomic_info)

        feature_names += element_features
        feature_values += element_values

        features = namedtuple(
            "MolecularInformation", feature_names
        )

        return features(*feature_values)

    def get_elemental_profile(self, molecule=None, atomic_info=None, normalize=True):
        if atomic_info is None:
            atomic_info = self.get_elements_info(molecule=molecule)
        unique_elements = self.get_unique_elements(atomic_info=atomic_info)

        element_features = list()
        element_values = list()
        molar_mass = self.get_molar_mass(atomic_info=atomic_info)

        for element in unique_elements:
            element_features.append(f"{element.lower()}_count")
            element_count = self.get_element_frequency(element=element, molecule=molecule, atomic_info=atomic_info)
            element_values.append(element_count)

            element_features.append(f"{element.lower()}_total_mass")
            element_mass = self.get_total_element_mass(element=element, molecule=molecule, atomic_info=atomic_info)
            element_values.append(element_mass)

            if normalize:
                element_features.append(f"{element.lower()}_count_proportion")
                element_values.append(element_count / len(atomic_info))

                element_features.append(f"{element.lower()}_mass_proportion")
                element_values.append(
                    self.get_total_element_mass(element=element, atomic_info=atomic_info) / molar_mass
                )

        bond_types, bond_counts = zip(*self.get_bond_distribution(molecule).items())
        element_features += list(map(lambda x: f"num_{x.lower()}_bonds", bond_types))
        element_values += bond_counts

        element_features.append("num_hydrogen_donors")
        element_features.append("num_hydrogen_acceptors")

        num_hydrogen_donors = self.count_hydrogen_donors(molecule)
        num_hydrogen_acceptors = self.count_hydrogen_acceptors(molecule)

        element_values.append(num_hydrogen_donors)
        element_values.append(num_hydrogen_acceptors)

        return element_features, element_values

    def featurize(self, molecules=None, molecular_info=None):
        if isinstance(molecules, list):
            features = self.batch_featurize(molecules=molecules, molecular_info_dump=molecular_info)
            features = features[0] if len(features) == 1 else features
        else:
            features = self.stream_featurize(molecule=molecules, atomic_info=molecular_info)

        return features

    def batch_featurize(self, molecules=None, molecular_info_dump=None):
        if molecular_info_dump is None:
            molecular_info_dump = [
                self.get_elements_info(molecule=molecule) for molecule in molecules
            ]
        return [
            self.stream_featurize(molecule=molecule, atomic_info=molecular_info)
            for molecule, molecular_info in zip(molecules, molecular_info_dump)
        ]

    def text_featurize(self, molecule):
        return None

    def batch_text_featurize(self, molecules):
        return None

    def _to_selfies(self, molecule):
        repr_kind = molecule.repr_string
        repr_type = molecule.repr_type

        if repr_type == "selfies":
            return molecule
        else:
            if repr_type == "inchi":
                repr_kind = Chem.MolToSmiles(molecule.rdkit_mol)
                repr_kind = encoder(repr_kind)

        return Molecule(repr_kind, "selfies")



if __name__ == "__main__":
    prob = 0.1

    #periodic_table = rdkit.Chem.GetPeriodicTable()

    featurizer = MoleculeFeaturizer()

    if prob > 0.5:
        inchi = "InChI=1S/C6H5NO2/c8-7(9)6-4-2-1-3-5-6/h1-5H"
        smiles = "CCC(Cl)C=C"
        selfies_form = encoder(smiles)
        repr_type = "inchi"
        mol = Molecule(inchi, repr_type)

        mol_info = featurizer.get_elements_info(molecule=mol)

        print(f"SMILES: {smiles}")
        print(f"SMILES -> SELFIES -> SMILES: {decoder(encoder(smiles))}")

        print(featurizer.count_bonds(mol, bond_type="SINGLE"))
        print(f"Molar mass by featurizer = {featurizer.get_molar_mass(atomic_info=mol_info)}")
        print(f"Bond distribution: {featurizer.get_bond_distribution(molecule=mol,)}")
        print(f"Bond types: {featurizer.get_unique_bond_types(molecule=mol, )}")
        print(featurizer.featurize(mol))
        print("Information on all elements in molecule: ",
            featurizer.get_elements_info(
                mol,
            )
        )

    else:
        molecular_info = {
            "InChI=1S/C6H5NO2/c8-7(9)6-4-2-1-3-5-6/h1-5H": "inchi",
            "CCC(Cl)C=C": "smiles",
            encoder("CCC(Cl)C=C"): "selfies",
        }

        mols = [Molecule(k, v) for k, v in molecular_info.items()]

        print(featurizer.featurize(molecules=mols))
        index = random.randint(0, len(mols)-1)
        print(f"Molecule {index} is represented by: ", mols[index].repr_string)

        element = "n"
        print(
            f"Element {element} appears {featurizer.get_element_frequency(molecule=mols[index],element=element)} times"
        )
        print(mols[index].get_name())

        print(f"Molecule {index} has {featurizer.count_rotable_bonds(mols[index])} rotable bonds.")
