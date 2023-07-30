# -*- coding: utf-8 -*-

"""Abstract base class and wrappers for featurizers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np
from scipy.spatial import distance_matrix

import rdkit

from chemcaption.featurize.text import Prompt
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule

# Implemented abstract and high-level classes

__all__ = [
    "AbstractFeaturizer",  # Featurizer base class.
    "MultipleFeaturizer",  # Combines multiple featurizers.
    "RDKitAdaptor",  # Higher-level featurizer. Returns lower-level featurizer instances.
    "Comparator",  # Class for comparing featurizer results amongst molecules.
]


"""Abstract class"""


class AbstractFeaturizer(ABC):
    """Abstract base class for lower level Featurizers."""

    def __init__(self):
        """Initialize class. Initialize periodic table."""
        self.periodic_table = rdkit.Chem.GetPeriodicTable()
        self._label = list()
        self.template = None

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

    def text_featurize(
        self,
        molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule],
    ) -> Prompt:
        """Embed features in Prompt instance.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

        Returns:
            (Prompt): Instance of Prompt containing relevant information extracted from `molecule`.
        """
        completion = self.featurize(molecule=molecule).tolist()
        completion = completion[0] if len(completion) == 1 else completion

        completion_type = (
            [type(c) for c in completion] if isinstance(completion, list) else type(completion)
        )

        representation = molecule.representation_string
        representation_type = molecule.__repr__().split("Mole")[0]

        completion_names = self.feature_labels()
        completion_names = completion_names[0] if len(completion_names) == 0 else completion_names

        return Prompt(
            completion=completion,
            completion_type=completion_type,
            representation=representation,
            representation_type=representation_type,
            completion_names=completion_names,
            template=self.template,
        )

    def text_featurize_many(
        self,
        molecules: Sequence[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]],
    ) -> List[Prompt]:
        """Embed features in Prompt instance for multiple molecules.

        Args:
            molecules (Sequence[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]]):
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


"""Higher-level featurizers."""


class MultipleFeaturizer(AbstractFeaturizer):
    """A featurizer to combine featurizers."""

    def __init__(self, featurizers: List[AbstractFeaturizer]):
        """Initialize class instance.

        Args:
            featurizers (List[AbstractFeaturizer]): A list of featurizer objects.

        """
        super().__init__()
        self.featurizers = featurizers

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Featurize a molecule instance. Returns results from multiple lower-level featurizers.

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

    def fit_on_featurizers(self, featurizers: List[AbstractFeaturizer]):
        """Fit MultipleFeaturizer instance on lower-level featurizers.

        Args:
            featurizers (List[AbstractFeaturizer]): List of lower-level featurizers.

        Returns:
            self : Instance of self with state updated.
        """
        self.featurizers = featurizers
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


class RDKitAdaptor(AbstractFeaturizer):
    """Higher-level featurizer. Returns specific, lower-level featurizers."""

    def __init__(
        self, rdkit_function: Callable, labels: List[str], **rdkit_function_kwargs: Dict[str, Any]
    ):
        """Initialize class object.

        Args:
            rdkit_function (Callable): Molecule descriptor-generating function.
                May be obtained from a chemistry featurization package like `rdkit` or custom written.
            labels (List[str]): Feature label(s) to assign to extracted feature(s).
            rdkit_function_kwargs (Dict[str, Any]): Keyword arguments to be parsed by `rdkit_function`.
        """
        super().__init__()
        self.rdkit_function = rdkit_function
        self._labels = labels
        self.rdkit_function_kwargs = rdkit_function_kwargs

    def featurize(
        self,
        molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule],
    ) -> np.array:
        """
        Featurize single molecule instance. Extract and return features from molecular object.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

        Returns:
            (np.array): Array containing extracted features.
        """
        feature = self.rdkit_function(molecule.rdkit_mol, **self.rdkit_function_kwargs)
        feature = (
            [
                feature,
            ]
            if isinstance(feature, (int, float))
            else feature
        )
        return np.array(feature).reshape((1, -1))

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class Comparator(MultipleFeaturizer):
    """Compare molecules based on featurizer outputs."""

    def __init__(self, featurizers: List[AbstractFeaturizer]):
        """Instantiate class.

        Args:
            featurizers (List[AbstractFeaturizer]): List of featurizers to compare over.

        """
        super().__init__(featurizers=featurizers)

    def _compare_on_featurizer(
        self,
        featurizer: AbstractFeaturizer,
        molecules: List[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]],
        epsilon: float = 0.,
    ) -> np.array:
        """Return results of molecule feature comparison between molecule instance pairs.

        Args:
            featurizer (AbstractFeaturizer): Featurizer to compare on.
            molecules (List[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]]):
                List containing a pair of molecule instances.
            epsilon (float): Small float. Precision bound for numerical inconsistencies.

        Returns:
            (np.array): Comparison results. 1 if all extracted features are equal, else 0.
        """
        batch_results = featurizer.featurize_many(molecules=molecules)

        distance_results = distance_matrix(batch_results, batch_results)

        return (np.mean(distance_results) <= epsilon).astype(int).reshape((1, -1))

    def featurize(
        self, molecules: List[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]],
        epsilon: float = .0,
    ) -> np.array:
        """
        Featurize multiple molecule instances.

        Extract and return comparison between molecular instances. 1 if similar, else 0.

        Args:
            molecules (List[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies.

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
            None

        Returns:
            List[str]: List of labels for all features extracted by all featurizers.
        """
        labels = list()
        for featurizer in self.featurizers:
            labels += featurizer.feature_labels()

        labels = [
            (label + "_similarity" if not label.__contains__("similarity") else label)
            for label in labels
        ]

        return labels

    def compare(
        self, molecules: List[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]]
    ) -> np.array:
        """
        Compare features from multiple molecular instances. 1 if all molecules are similar, else 0.

        Args:
            molecules (List[Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]]): Molecule instances to be compared.

        Returns:
            (np.array): Array containing comparison results with shape `(1, N)`,
                where `N` is the number of featurizers provided at initialization time.
        """
        return self.featurize(molecules=molecules)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
