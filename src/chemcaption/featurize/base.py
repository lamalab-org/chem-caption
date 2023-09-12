# -*- coding: utf-8 -*-

"""Abstract base class and wrappers for featurizers."""

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Generator, List, Optional

import numpy as np
import pandas as pd
import rdkit
from scipy.spatial import distance_matrix

from chemcaption.featurize.text import Prompt
from chemcaption.featurize.utils import apply_featurizer, join_list_elements
from chemcaption.molecules import Molecule

# Implemented abstract and high-level classes

__all__ = [
    "AbstractFeaturizer",  # Featurizer base class.
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
            np.array: An array of features for each molecule instance.
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
            List[str]: List of implementors.
        """
        raise NotImplementedError

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
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
            List[str]: List of implementors.
        """
        raise NotImplementedError

    @abstractmethod
    def feature_labels(self) -> List[str]:
        """Return feature label.

        Args:
            None.

        Returns:
            (List[str]): List of names of extracted features.
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
            featurizers (Optional[List[AbstractFeaturizer]]): A list of featurizer objects. Defaults to `None`.

        """
        super().__init__()

        self.featurizers = None
        self.fit_on_featurizers(featurizers=featurizers)

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize a molecule instance. Returns results from multiple lower-level featurizers.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            features (np.array), array shape [1, num_featurizers]: Array containing features
                extracted from molecule.
                `num_featurizers` is the number of featurizers passed to MultipleFeaturizer.
        """
        featurizer_molecule_pairs = [(featurizer, molecule) for featurizer in self.featurizers]

        with ProcessPoolExecutor() as executor:
            features = list(executor.map(apply_featurizer, featurizer_molecule_pairs))

        return np.concatenate(features, axis=-1)

    def text_featurize(
        self,
        molecule: Molecule,
    ) -> Generator[Prompt, None, None]:
        """Embed features in Prompt instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (Prompt): Instance of Prompt containing relevant information extracted from `molecule`.
        """
        return (featurizer.text_featurize(molecule=molecule) for featurizer in self.featurizers)

    def feature_labels(self) -> List[str]:
        """Return feature labels.

        Args:
            None.

        Returns:
            List[str]: List of labels for all features extracted by all featurizers.
        """
        labels = list()
        for featurizer in self.featurizers:
            labels += featurizer.feature_labels()

        return labels

    def fit_on_featurizers(self, featurizers: Optional[List[AbstractFeaturizer]] = None):
        """Fit MultipleFeaturizer instance on lower-level featurizers.

        Args:
            featurizers (Optional[List[AbstractFeaturizer]]): List of lower-level featurizers. Defaults to `None`.

        Returns:
            self : Instance of self with state updated.
        """
        # Type check for AbstractFeaturizer instances
        for ix, featurizer in enumerate(featurizers):
            # Each featurizer must be specifically of type AbstractFeaturizer

            if not isinstance(featurizer, AbstractFeaturizer):
                raise ValueError(
                    f"`{featurizer.__class__.__name__}` instance at index {ix} is not of type `AbstractFeaturizer`."
                )

        self.featurizers = featurizers

        print(f"`{self.__class__.__name__}` instance fitted with {len(featurizers)} featurizers!\n")
        self.label = self.feature_labels()

        self.prompt_template = [featurizer.prompt_template for featurizer in featurizers]
        self.completion_template = [featurizer.completion_template for featurizer in featurizers]

        self._names = [featurizer._names for featurizer in featurizers]

        return self

    def generate_data(self, molecules: List[Molecule], metadata: bool = False) -> pd.DataFrame:
        """Convert generated feature array to DataFrame.

        Args:
            molecules (List[Molecule]): Collection of molecular instances.
            metadata (bool): Include extra molecule information.

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

    def batch_generate_data(
        self, molecules: List[Molecule], batch_size: int = 2, fpath: str = "molecule_features.csv"
    ) -> None:
        """Generate and persist molecular features in batches.

        Args:
            molecules (List[Molecule]): Collection of molecular instances.
            batch_size (int): Number of batches to extract at a time.
            fpath (str): Path to persisted data.

        Returns:
            None.
        """
        if not os.path.isdir(fpath):
            raise FileExistsError(
                "Value of `fpath` parameter is a directory. Specify a file handle.`"
            )

        # Split molecules into batches
        molecule_batches = [
            molecules[i : i + batch_size] for i in range(0, len(molecules), batch_size)
        ]
        num_batches = len(molecule_batches)

        for i, molecule_batch in enumerate(molecule_batches, start=1):
            molecule_features = self.generate_data(molecules=molecule_batch)

            if os.path.isfile(fpath):
                try:
                    source_of_truth = pd.read_csv(fpath)
                    molecule_features = pd.concat((source_of_truth, molecule_features), axis=0)
                except:
                    pass

            molecule_features.to_csv(path_or_buf=fpath, index=False)
            print(f"Data batch persisted ({i}/{num_batches})!\n")

        print("=" * 30, f"\nAll data persisted ({num_batches}/{num_batches})!")

        return None

    def generate_text_data(self, molecules: List[Molecule]):
        """Convert generated feature array to text and embed in PromptContainer.

        Args:
            molecules (List[Molecule]): Collection of molecular instances.

        Returns:
            (PromptContainer): PromptContainer generated from feature array.
        """
        raise NotImplementedError

    def batch_generate_text_data(
        self, molecules: List[Molecule], batch_size: int = 2, fpath: str = "molecule_features.jsonl"
    ) -> None:
        """Generate and persist generated PromptContainer object.

        Args:
            molecules (List[Molecule]): Collection of molecular instances.
            batch_size (int): Number of batches to extract at a time.
            fpath (str): Path to persisted data.

        Returns:
            None.
        """
        raise NotImplementedError

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
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
        """Return feature labels.

        Args:
            None.

        Returns:
            List[str]: List of labels for all features extracted by all featurizers.
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
            List[str]: List of implementors.
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
            self : Instance of self with state updated.
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
        """Return feature labels.

        Args:
            None.

        Returns:
            List[str]: List of labels for all features extracted by all comparators.
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
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
