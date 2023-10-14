# -*- coding: utf-8 -*-

"""Abstract base class and wrappers for featurizers."""

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rdkit
from colorama import Fore
from frozendict import frozendict
from morfeus import SASA, XTB, read_xyz
from rdkit import Chem
from scipy.spatial import distance_matrix

from chemcaption.featurize.text import Prompt, PromptCollection
from chemcaption.featurize.utils import cached_conformer, join_list_elements
from chemcaption.molecules import Molecule, SMILESMolecule

# Implemented abstract and high-level classes

__all__ = [
    "AbstractFeaturizer",  # Featurizer base class.
    "MorfeusFeaturizer",
    "AbstractComparator",
    "MultipleFeaturizer",  # Combines multiple featurizers.
    "Comparator",  # Class for comparing featurizer results amongst molecules.
    "MultipleComparator",  # Higher-level Comparator. Returns lower-level Comparator instances.
    "PERIODIC_TABLE",  # Periodic table
]

PERIODIC_TABLE = rdkit.Chem.GetPeriodicTable()  # Periodic table


"""Abstract class"""


class AbstractFeaturizer(ABC):
    """Abstract base class for lower level Featurizers."""

    def __init__(self):
        """Initialize class. Initialize periodic table."""
        self.prompt_template = "Question: What is the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} {REPR_STRING}?"
        self.completion_template = "Answer: {COMPLETION}"
        self._names = []
        self.constraint = None

    def get_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        return self._names

    @abstractmethod
    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single Molecule instance."""
        raise NotImplementedError

    def featurize_many(self, molecules: List[Molecule]) -> np.array:
        """
        Featurize a sequence of Molecule objects.

        Args:
            molecules (Sequence[Molecule]):
                A sequence of molecule representations.

        Returns:
            (np.array): An array of features for each molecule instance.
        """
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.featurize, molecules))

        return np.concatenate(results)

    def text_featurize(
        self,
        molecule: Molecule,
    ) -> Prompt:
        """Embed features in Prompt instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (Prompt): Instance of Prompt containing relevant information extracted from `molecule`.
        """
        completion = self.featurize(molecule=molecule).tolist()

        completion_type = [type(c) for c in completion]
        representation = molecule.representation_string
        representation_type = molecule.__repr__().split("Mole")[0]

        completion_labels = self.feature_labels()

        completion_name = self.get_names()[0]["noun"]

        return Prompt(
            completion=join_list_elements(completion),
            completion_type=completion_type,
            representation=representation,
            representation_type=representation_type,
            completion_names=completion_name,
            completion_labels=completion_labels,
            prompt_template=self.prompt_template,
            completion_template=self.completion_template,
            constraint=self.constraint,
        )

    def text_featurize_many(
        self,
        molecules: List[Molecule],
    ) -> List[Prompt]:
        """Embed features in Prompt instance for multiple molecules.

        Args:
            molecules (Sequence[Molecule]):
                A sequence of molecule representations.

        Returns:
            (List[Prompt]): List of Prompt instances containing relevant information extracted from each
                molecule in `molecules`.
        """
        return [self.text_featurize(molecule=molecule) for molecule in molecules]

    @abstractmethod
    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        raise NotImplementedError

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
        raise NotImplementedError

    def feature_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        return self._names

    def citations(self):
        """Return citation for this project."""
        raise NotImplementedError


class MorfeusFeaturizer(AbstractFeaturizer):
    """Abstract featurizer for morfeus-generated features."""

    def __init__(
        self,
        file_name: Optional[str] = None,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate class.

        Args:
            file_name (Optional[str]): Name for temporary XYZ file.
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
        """
        super().__init__()
        self._conf_gen_kwargs = (
            frozendict(conformer_generation_kwargs)
            if conformer_generation_kwargs
            else frozendict({})
        )
        self.morfeus_kwargs = frozendict(morfeus_kwargs) if morfeus_kwargs else frozendict({})

        if file_name is None:
            file_name = self._get_random_file_name()

        self.random_file_name = file_name if file_name.endswith(".xyz") else file_name + ".xyz"

    @staticmethod
    def _get_random_file_name() -> str:
        """Generate a random file name.

        Args:
            None.

        Returns:
            (str): Randomly generated filename.
        """
        num = np.random.randint(low=1, high=100)
        numbers = np.random.randint(low=65, high=82, size=(num,)).flatten().tolist()
        letters = list(map(lambda x: chr(x), numbers))

        file_name = "".join(letters) + ".xyz"

        return file_name

    def _get_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        """Return conformer for molecule.

        Args:
            mol (Chem.Mol): rdkit Molecule.

        Returns:
            (Chem.Mol): Molecule instance embedded with conformers.
        """
        smiles = Chem.MolToSmiles(mol)
        return cached_conformer(smiles, self._conf_gen_kwargs)

    def _mol_to_xyz_file(self, molecule: Molecule) -> None:
        """Generate XYZ block from molecule instance.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            None.
        """
        mol = molecule.rdkit_mol
        mol = self._get_conformer(mol)

        Chem.rdmolfiles.MolToXYZFile(mol, self.random_file_name)
        return

    def _xyz_file_to_mol(self) -> SMILESMolecule:
        """Generate XYZ block from molecule instance.

        Args:
            None.

        Returns:
            (SMILESMolecule): SMILES molecule instance.
        """
        mol = Chem.rdmolfiles.MolFromXYZFile(self.random_file_name)
        smiles = Chem.MolToSmiles(mol)

        return SMILESMolecule(smiles)

    @staticmethod
    def _parse_indices(
        atom_indices: Union[int, List[int]], as_range: bool = False
    ) -> Tuple[Sequence, bool]:
        """Preprocess indices.

        Args:
            atom_indices (Union[int, List[int]]): Range of atoms to calculate areas for. Either:
                - an integer,
                - a list of integers, or
                - a two-tuple of integers representing lower index and upper index.
            as_range (bool): Use `atom_indices` parameter as a range of indices or not. Defaults to `False`
        """
        if as_range:
            if isinstance(atom_indices, int):
                atom_indices = range(1, atom_indices + 1)

            elif len(atom_indices) == 2:
                if atom_indices[0] > atom_indices[1]:
                    raise IndexError(
                        "`atom_indices` parameter should contain two integers as (lower, upper) i.e., [10, 20]"
                    )
                atom_indices = range(atom_indices[0], atom_indices[1] + 1)

            else:
                as_range = False
                print(
                    Fore.RED
                    + "UserWarning: List of integers passed to `atom_indices` parameter. `as_range` parameter will be refactored to False."
                    + Fore.RESET
                )

        else:
            if isinstance(atom_indices, int):
                atom_indices = [atom_indices]

        return atom_indices, as_range

    def _get_element_coordinates(
        self, molecule: Molecule
    ) -> Tuple[List[Union[int, str]], np.array]:
        """Return appropriate morfeus instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (Tuple[List[Union[int, str]], np.array]): Tuple containing
                - elements in molecule and
                - their corresponding coordinates.
        """
        self._mol_to_xyz_file(molecule)  # Persist molecule in XYZ file
        elements, coordinates = read_xyz(self.random_file_name)  # Read file

        os.remove(self.random_file_name)  # Eliminate file

        return elements, coordinates

    def _get_morfeus_instance(
        self, molecule: Molecule, morpheus_instance: str = "xtb"
    ) -> Union[SASA, XTB]:
        """Return appropriate morfeus instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.
            morpheus_instance (str): Type of morfeus instance. Can take on either `xtb` or `sasa`. Defaults to `xtb`.

        Returns:
            (Union[SASA, XTB]): Appropriate morfeus instance.
        """
        if morpheus_instance.lower() not in ["xtb", "sasa"]:
            raise Exception(
                "`morpheus_instance` parameter must take on either `xtb` or `sasa` as value."
            )

        return (
            self._get_sasa_instance(molecule)
            if morpheus_instance.lower() == "sasa"
            else self._get_xtb_instance(molecule)
        )

    def _get_xtb_instance(self, molecule: Molecule) -> XTB:
        """Return appropriate morfeus instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (XTB): Appropriate morfeus XTB instance.
        """
        elements, coordinates = self._get_element_coordinates(molecule)

        return XTB(elements, coordinates, "1")

    def _get_sasa_instance(self, molecule: Molecule) -> SASA:
        """Return appropriate morfeus instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (SASA): Appropriate morfeus SASA instance.
        """
        elements, coordinates = self._get_element_coordinates(molecule)

        return SASA(elements, coordinates, **self.morfeus_kwargs)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class AbstractComparator(ABC):
    """Abstract base class for Comparator objects."""

    def __init__(self):
        """Initialize class. Initialize periodic table."""
        self.template = None
        self._names = []

    @abstractmethod
    def featurize(self, molecules: List[Molecule]) -> np.array:
        """Featurize multiple Molecule instances."""
        raise NotImplementedError

    @abstractmethod
    def compare(self, molecules: List[Molecule]) -> np.array:
        """Compare features from multiple molecular instances. 1 if all molecules are similar, else 0."""
        raise NotImplementedError

    @abstractmethod
    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        raise NotImplementedError

    @abstractmethod
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
        raise NotImplementedError

    def citations(self):
        """Return citation for this project."""
        return None


"""Higher-level featurizers."""


class MultipleFeaturizer(AbstractFeaturizer):
    """A featurizer to combine featurizers."""

    def __init__(self, featurizers: Optional[List[AbstractFeaturizer]] = None):
        """Initialize class instance.

        Args:
            featurizers (Optional[List[AbstractFeaturizer]]):
                A list of featurizer objects. Defaults to `None`.

        """
        super().__init__()

        self.featurizers = featurizers

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize a molecule instance.
        Returns results from multiple lower-level featurizers.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            features (np.array), array shape [1, num_featurizers]:
                Array containing features extracted from molecule.
                `num_featurizers` is the number of featurizers passed to MultipleFeaturizer.
        """
        features = [
            feature for f in self.featurizers for feature in f.featurize(molecule).flatten()
        ]

        return np.array(features).reshape((1, -1))

    def text_featurize(
        self,
        molecule: Molecule,
    ) -> PromptCollection:
        """Embed features in Prompt instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (PromptCollection): Instance of Prompt containing relevant information extracted from `molecule`.
        """
        return PromptCollection([f.text_featurize(molecule=molecule) for f in self.featurizers])

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels for all features extracted by all featurizers.
        """
        labels = [label for f in self.featurizers for label in f.feature_labels()]

        return labels

    def generate_data(self, molecules: List[Molecule], metadata: bool = False) -> pd.DataFrame:
        """Convert generated feature array to DataFrame.

        Args:
            molecules (List[Molecule]): Collection of molecular instances.
            metadata (bool): Include extra molecule information.
                Defaults to `False`.

        Returns:
            (pd.DataFrame): DataFrame generated from feature array.
        """
        features = self.featurize_many(molecules=molecules)

        if metadata:
            extra_columns = ["representation_system", "representation_string"]
            extra_features = np.array(
                [[mol.get_representation(), mol.representation_string] for mol in molecules]
            ).reshape((-1, 2))

            features = np.concatenate([extra_features, features], axis=-1)
        else:
            extra_columns = []

        columns = extra_columns + self.feature_labels()

        return pd.DataFrame(data=features, columns=columns)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class Comparator(AbstractComparator):
    """Compare molecules based on featurizer outputs."""

    def __init__(self, featurizers: Optional[List[AbstractFeaturizer]] = None):
        """Instantiate class.

        Args:
            featurizers (Optional[List[AbstractFeaturizer]]): List of featurizers to compare over. Defaults to `None`.

        """
        super().__init__()
        self.featurizers = None
        self.fit_on_featurizers(featurizers=featurizers)

    def fit_on_featurizers(self, featurizers: Optional[List[AbstractFeaturizer]] = None):
        """Fit Comparator instance on lower-level featurizers.

        Args:
            featurizers (Optional[List[AbstractFeaturizer]]): List of lower-level featurizers. Defaults to `None`.

        Returns:
            self : Instance of self with state updated.
        """
        if featurizers is None:
            self.featurizers = featurizers
            return self
        # Type check for AbstractFeaturizer instances
        for ix, featurizer in enumerate(featurizers):
            # Each featurizer must be specifically of type AbstractFeaturizer

            if not isinstance(featurizer, AbstractFeaturizer):
                raise ValueError(
                    f"`{featurizer.__class__.__name__}` instance at index {ix} is not of type `AbstractFeaturizer`."
                )

        self.featurizers = featurizers

        return self

    def __str__(self):
        """Return string representation.

        Args:
            None.

        Returns:
            (str): String representation of `self`.
        """
        return self.__class__.__name__

    def _compare_on_featurizer(
        self,
        featurizer: AbstractFeaturizer,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.array:
        """Return results of molecule feature comparison between molecule instance pairs.

        Args:
            featurizer (AbstractFeaturizer): Featurizer to compare on.
            molecules (List[Molecule]):
                List containing a pair of molecule instances.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            (np.array): Comparison results. 1 if all extracted features are equal, else 0.
        """
        batch_results = featurizer.featurize_many(molecules=molecules)

        distance_results = distance_matrix(batch_results, batch_results)

        return (np.mean(distance_results) <= epsilon).astype(int).reshape((1, -1))

    def featurize(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.array:
        """
        Featurize multiple molecule instances.

        Extract and return comparison between molecular instances. 1 if similar, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            (np.array): Array containing extracted features with shape `(1, N)`,
                where `N` is the number of featurizers provided at initialization time.
        """
        results = [
            self._compare_on_featurizer(featurizer=featurizer, molecules=molecules, epsilon=epsilon)
            for featurizer in self.featurizers
        ]
        return np.concatenate(results, axis=-1)

    def feature_labels(
        self,
    ) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for all features extracted by all featurizers.
        """
        labels = []
        for featurizer in self.featurizers:
            labels += featurizer.feature_labels()

        labels = [label + "_similarity" for label in labels]

        return labels

    def compare(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.array:
        """
        Compare features from multiple molecular instances. 1 if all molecules are similar, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            (np.array): Array containing comparison results with shape `(1, N)`,
                where `N` is the number of featurizers provided at initialization time.
        """
        return self.featurize(molecules=molecules, epsilon=epsilon)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MultipleComparator(Comparator):
    """A Comparator to combine Comparators."""

    def __init__(self, comparators: Optional[List[Comparator]] = None):
        """Instantiate class.

        Args:
            comparators (Optional[List[Comparator]]): List of Comparator instances. Defaults to `None`.
        """
        super().__init__()

        self.comparators = None

        self.fit_on_comparators(comparators=comparators)  # If all comparators pass the check

    def fit_on_comparators(self, comparators: Optional[List[Comparator]] = None):
        """Fit MultipleComparator instance on lower-level comparators.

        Args:
            comparators (Optional[List[Comparator]]): List of lower-level comparators. Defaults to `None`.

        Returns:
            (self) : Instance of self with state updated.
        """
        if comparators is None:
            self.comparators = comparators
            return self

        # Type check for Comparator instances
        for ix, comparator in enumerate(comparators):
            # Each comparator must be specifically of types:
            #   `Comparator` or `MultipleComparator` (allows for nesting purposes)

            if not isinstance(comparator, Comparator):
                raise ValueError(
                    f"`{comparator.__class__.__name__}` instance at index {ix}",
                    " is not of type `Comparator` or `MultipleComparator`.",
                )
        self.comparators = comparators

        return self

    def featurize(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.array:
        """
        Compare features from multiple molecular Comparators. 1 if all molecules are similar, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            (np.array): Array containing comparison results with shape `(1, N)`,
                where `N` is the number of Comparators provided at initialization time.
        """
        features = [
            comparator.featurize(molecules=molecules, epsilon=epsilon)
            for comparator in self.comparators
        ]

        return np.concatenate(features, axis=-1)

    def feature_labels(
        self,
    ) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for all features extracted by all comparators.
        """
        labels = []
        for comparator in self.comparators:
            labels += comparator.feature_labels()

        return labels

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
