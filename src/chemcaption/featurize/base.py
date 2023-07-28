"""Abstract base class and wrappers for featurizers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import rdkit
from rdkit.Chem import Descriptors, rdMolDescriptors

from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule
from chemcaption.presets import inspect_info


class AbstractFeaturizer(ABC):
    """Base class for lower level Featurizers."""

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
