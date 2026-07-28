# -*- coding: utf-8 -*-

"""Featurizers describing the composition of a molecule."""

from collections import Counter
from typing import Dict, List, Optional, Union

import numpy as np
from rdkit.Chem import Descriptors

from chemcaption.featurize.base import PERIODIC_TABLE, AbstractFeaturizer
from chemcaption.featurize.utils import answer_generation
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

    def __init__(self, completion_template: Optional[str] = None, version: bool = 0):
        """
        Initialize class.

        Args:
            completion_template (Optional[str]): Custom completion template, Defaults to a descriptive template with SMILES and property name.
            version (int): Index into COMPLETION_TEMPLATES. Defaults to 0.
        """
        super().__init__(completion_template=completion_template, version=version)
        
        self._names = [
            {
                "noun": "molecular formula",
            }
        ]
    
    def get_completion_template(self, version: bool = 0) -> str:
        templates = [
            "The {PROPERTY_NAME} of the molecule {VERB} {PROPERTY_VALUE}.",
            "The molecule has a {PROPERTY_NAME} of {PROPERTY_VALUE}.",
        ]
        return templates[version]

    @property
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels for extracted features.
        """
        return ["molecular_formula"]

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize single molecule instance. Get the molecular formular of a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            str: Molecular formular of `molecule`.
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

    def __init__(self, completion_template: Optional[str] = None, version: bool = 0):
        """
        Get the molecular mass of a molecule.
        
        Args:
            completion_template (Optional[str]): Custom completion template. Defaults to descriptive template with SMILES and property name.
        """
        super().__init__(completion_template=completion_template, version=version)

        self.template = (
            "What {VERB} the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "molecular mass",
            }
        ]
        
    def get_completion_template(self, version: bool = 0) -> str:
        templates = [
            "The {PROPERTY_NAME} of the molecule {VERB} {PROPERTY_VALUE}.",
            "The molecule has a {PROPERTY_NAME} of {PROPERTY_VALUE}."
        ]
        return templates[version]

    @property
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
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
            float: Molecular mass of `molecule`.
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

    def __init__(self, completion_template: Optional[str] = None, version: bool = 0):
        """Instantiate instance."""
        super().__init__(completion_template=completion_template, version=version)

        self.template = (
            "What {VERB} the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
        self._names = [
            {
                "noun": "monoisotopic molecular mass",
            }
        ]
    
    def get_completion_template(self, version: bool = 0):
        template = [
                "The {PROPERTY_NAME} of the molecule {VERB} {PROPERTY_VALUE}.",
                "For this molecule, the {PROPERTY_NAME} {VERB} {PROPERTY_VALUE}."
            ]
        return template[version]

    @property
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
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
            float: Monoisotopic molecular mass of `molecule`.
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

    def __init__(
        self,
        preset: Optional[Union[List[str], Dict[str, str]]] = None,
        completion_template: Optional[str] = None,
        version: bool = 0
    ):
        """Get the total mass component of an element in a molecule.

        Args:
            preset (Optional[Union[List[str], Dict[str, str]]]): Preset containing substances or elements of interest.
            completion_template (Optional[str]): Custom completion template. Defaults to base class template.
        """
        preset = (
            preset if preset is not None
            else ["Carbon", "Hydrogen", "Nitrogen", "Oxygen"]
        )

        super().__init__(preset=preset, completion_template=completion_template, version=version)

        self.template = (
            "What {VERB} the {PROPERTY_NAME} for the molecule with {REPR_SYSTEM} `{REPR_STRING}`?"
        )
    
    def get_completion_template(self, version: bool = 0):
        template = [
                "The {PROPERTY_NAME} in the molecule {VERB} {PROPERTY_VALUE}.",
                "The molecule has a {PROPERTY_NAME} of {PROPERTY_VALUE}."
            ]
        return template[version]

    @property
    def get_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        noun = "masses" if len(self.preset) > 1 else "mass"
        return [{"noun": f"total {noun} of " + answer_generation(self.preset)}]

    @property
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
        return [element.lower() + "_mass" for element in self.preset]

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
            ElementMassFeaturizer: Instance of self with updated state.
        """
        if isinstance(molecules, list):
            unique_elements = set()
            for molecule in molecules:
                unique_elements.update(set(self._get_unique_elements(molecule)))
        else:
            unique_elements = set(self._get_unique_elements(molecules))

        self.preset = list(unique_elements)

        return self

    @staticmethod
    def _get_element_mass(element: str, molecule: Molecule) -> float:
        """
        Get the total mass component of an element in a molecule.

        Args:
            element (str): String representing element name or symbol.
            molecule (Molecule): Molecular representation.

        Returns:
            float: Total mass accounted for by `element` in `molecule`.
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

    def _get_profile(self, molecule: Molecule) -> List:
        """Generate molecular profile based of preset attribute.

        Args:
            molecule (Molecule): Molecular representation instance.

        Returns:
            List: List of elemental masses.
        """
        element_masses = [
            self._get_element_mass(element=element, molecule=molecule) for element in self.preset
        ]

        return element_masses

    @staticmethod
    def _get_unique_elements(molecule: Molecule) -> List[str]:
        """
        Get unique elements that make up a molecule.

        Args:
            molecule (Molecule): Molecular representation.

        Returns:
            List[str]: Unique list of element_names or element_symbols in `molecule`.
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

    def __init__(
        self,
        preset: Optional[List[str]] = None,
        completion_template: Optional[str] = None,
        version: bool = 0):
        """Initialize instance."""
        super().__init__(preset=preset, completion_template=completion_template, version=version)
        self.prefix = ""
        self.suffix = "_mass_ratio"
    
    def get_completion_template(self, version: bool = 0) -> str:
        template = [
            "The {PROPERTY_NAME} in the molecule {VERB} {PROPERTY_VALUE}.",
            "The molecule has a {PROPERTY_NAME} of {PROPERTY_VALUE}."
        ]
        return template[version]

    @property
    def get_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        proportion = "proportions" if len(self.preset) > 1 else "proportion"
        return [{"noun": f"mass {proportion} of " + answer_generation(self.preset)}]

    @property
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

    def __init__(
        self,
        preset: Optional[List[str]] = None,
        completion_template: Optional[str] = None,
        version: bool = 0,
        verbose_absent: bool = False,
        skip_zero: bool = False,
    ):
        """
        Get the count for each element in a molecule.

        Args:
            preset (Optional[List[str]]): Elements of interest. Defaults to None.
            completion_template (Optional[str]): Custom completion template.
            verbose_absent (bool): If True, render absent elements as "no Xs present". Defaults to False.
            skip_zero (bool): If True, omit elements with zero count from the description. Defaults to False.
        """
        super().__init__(preset=preset, completion_template=completion_template, version=version)
        self.verbose_absent = verbose_absent
        self.skip_zero = skip_zero
        self.smart_names = self.preset
    
    def get_completion_template(self, version: bool = 0) -> str:
        template = [
            "The molecule has {PROPERTY_VALUE}.",
            "There {VERB} {PROPERTY_VALUE} in the molecule."
        ]
        return template[version]
    
    @property
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
        return ["num_" + element.lower() + "_atoms" for element in self.preset]

    @property
    def get_names(self):
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        count = "counts" if len(self.preset) > 1 else "count"
        return [{"noun": f"atom {count} of " + answer_generation(
            self.preset, verbose_absent=self.verbose_absent, skip_zero=self.skip_zero
        )}]

    @staticmethod
    def _get_atom_count(element: str, molecule: Molecule) -> int:
        """
        Get number of atoms of element in a molecule.

        Args:
            element (str): String representation of a chemical element.
            molecule (Molecule): Molecular representation instance.

        Returns:
            int: Number of atoms of element in molecule.
        """

        atom_count = []

        for atom in molecule.get_atoms():
            if PERIODIC_TABLE.GetElementName(atom.GetAtomicNum()) == element:
                atom_count.append(atom)
            if PERIODIC_TABLE.GetElementSymbol(atom.GetAtomicNum()) == element:
                atom_count.append(atom)

        return len(atom_count)

    def _get_profile(self, molecule: Molecule) -> List:
        """Generate number of atoms per element based of preset attribute.

        Args:
            molecule (Molecule): Molecular representation instance.

        Returns:
            List: List of elemental atom counts.
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

    def __init__(self, completion_template: Optional[str] = None, preset: Optional[List[str]] = None, version: bool = 0):
        """Initialize instance.

        Args:
            preset (Optional[List[str]]): None or List of strings. Containing the names of elements of interest.
                Defaults to `None`.
        """
        super().__init__(preset=preset, completion_template=completion_template, version=version)
        self.smart_names = None

    def get_completion_template(self, version: bool = 0) -> str:
        template = [
            "The {PROPERTY_NAME} of the molecule {VERB} {PROPERTY_VALUE}.",
            "In this molecule, {PROPERTY_NAME} {VERB} {PROPERTY_VALUE}"
        ]
        return template[version]
    
    @property
    def get_names(self):
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        count = "counts" if len(self.preset) > 1 else "count"
        return [{"noun": f"relative atom {count} of " + answer_generation(
            self.preset, verbose_absent=self.verbose_absent, skip_zero=self.skip_zero
        )}]

    @property
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
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

    def __init__(self, completion_template: Optional[str] = None, preset: Optional[List[str]] = None, version: bool = 0):
        """Initialize instance."""
        super().__init__(completion_template=completion_template, preset=preset, version=version)
        self.smart_names = None
        self._names = [
            {
                "noun": "total number of atoms",
            }
        ]

    def get_completion_template(self, version: bool = 0) -> str:
        template = [
            "The molecule has a total of {PROPERTY_VALUE} atoms.",
            "The {PROPERTY_NAME} of the molecule {VERB} {PROPERTY_VALUE}."
        ]
        return template[version]
    
    @property
    def get_names(self):
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        return [{"noun": "total number of atoms"}]

    @property
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
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

    def __init__(self, completion_template: Optional[str] = None, preset: Optional[List[str]] = None,  version: bool = 0):
        """Instantiate class.

        Args:
            completion_template (Optional[str]): Custom completion template. Defaults to a descriptive template with SMILES and property name.
        """
        
        super().__init__(completion_template=completion_template, preset=preset, version=version)
        self._names = [
            {
                "noun": "degree of unsaturation",
            }
        ]

    def get_completion_template(self, version: bool = 0) -> str:
        template = [
            "The {PROPERTY_NAME} of the molecule {VERB} {PROPERTY_VALUE}.",
            "The molecule contains {PROPERTY_VALUE} unsaturated bond(s) in total."
        ]
        return template[version]

    @property
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
        return ["degree_of_unsaturation"]

    @staticmethod
    def _get_degree_of_unsaturation_for_mol(molecule: Molecule):
        """Return the degree of unsaturation for a molecule.

        .. math::
            {\\displaystyle \\mathrm {DU} =1+{\tfrac {1}{2}}\\sum n_{i}(v_{i}-2)}

        where ni is the number of atoms with valence vi.

        Args:
            molecule (Molecule): Molecule instance.

        Returns:
            int: Degree of unsaturation.
        """
        # add hydrogens
        mol = molecule.reveal_hydrogens()
        valence_counter: Counter = Counter()
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
