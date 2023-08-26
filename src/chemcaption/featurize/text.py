# -*- coding: utf-8 -*-

"""Classes for representing featurizer output as text."""

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from chemcaption.featurize.text_utils import inspect_info, generate_template
from chemcaption.molecules import Molecule

# Implemented text-related classes

__all__ = [
    "Prompt",
    "PromptContainer",
]


@dataclass
class Prompt:
    """Encapsulate all things prompt-related."""

    completion: Union[str, float, int, bool, List[Union[str, float, int, bool]]]
    representation: Union[str, List[str]]
    representation_type: Union[str, float, int, bool, np.array]
    completion_type: Union[str, float, int, bool, np.array]
    completion_names: Union[str, List[str]]
    completion_labels: Union[str, List[str]]
    template: Optional[str] = None

    def __post_init__(self):
        self.answer_template = generate_template(
            template_type="text",
            key="single" if len(self.completion) < 2 else "multiple"
        )

    def __dict__(self) -> Dict[str, Any]:
        """Return dictionary representation of object.

        Args:
            None.

        Returns:
            (dict): Dictionary containing all relevant prompt-related information.
        """
        return {
            "representation": self.representation,
            "representation_type": self.representation_type,
            "prompt": self.template,
            "completion": self.completion,
            "completion_names": self.completion_names,
            "completion_labels": self.completion_labels,
            "filled_prompt": self.fill_template(template="q"),
            "filled_completion": self.fill_template(template="a"),
        }

    def fill_template(self, template: str = "q", precision_type: str = "decimal") -> str:
        """Fill up the prompt template with appropriate values.

        Args:
            template (str): Type of template to fill. May be `q` or `a` fo `question` and `answer` respectively.
            precision_type (str): Level of precision for approximation purposes. Can be `decimal` or `significant`.
                Defaults to `decimal`.

        Returns:
            (str): Appropriately formatted template.
        """
        molecular_info = dict(
            PROPERTY_NAME=self.completion_names,
            REPR_SYSTEM=self.representation_type,
            REPR_STRING=self.representation,
            PROPERTY_VALUE=self.completion,
            PRECISION=4,
            PRECISION_TYPE=precision_type,
        )
        molecular_info = inspect_info(molecular_info)

        return (
            self.template.format(**molecular_info) if template == "q" else
            self.answer_template.format(**molecular_info)
        )

    def __str__(self) -> str:
        """Return string representation of object.

        Args:
            None.

        Returns:
            (str): Appropriately formatted template.
        """
        return self.fill_template(template="a")

    def to_meta_yaml(self):
        """Convert all prompt information from string to YAML format."""
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


class PromptContainer:
    """Contain Prompt object generators."""

    def __init__(
        self, prompt_iterable: Optional[Tuple[str, Generator[Prompt, None, None]]] = None
    ):
        """Initialize instance.

        Args:
            prompt_iterable (Optional[List[Tuple[str, Generator[Prompt, None, None]]]]):
                List of tuples/lists containing:
                    - Molecular string and
                    - generator for Prompt objects.

        """
        self.db = [{prompt_iterable[0]: prompt_iterable[1]}] if prompt_iterable is not None else []

    def __add_iter__(
        self, new_prompt_iterable: Tuple[str, Generator[Prompt, None, None]]
    ) -> None:
        """Store Prompt generator.

        Args:
            new_prompt_iterable (Tuple[str, Generator[Prompt, None, None]]):
                List of tuples/lists containing:
                    - Molecular string and
                    - generator for Prompt objects.
        """
        self.db.append({new_prompt_iterable[0]: new_prompt_iterable[1]})
        return

    def add(self, new_prompt_iterable: Tuple[str, Generator[Prompt, None, None]]) -> None:
        """Store Prompt generator.

        Args:
            new_prompt_iterable (Tuple[str, Generator[Prompt, None, None]]):
                List of tuples/lists containing:
                    - Molecular string and
                    - generator for Prompt objects.
        """
        self.__add_iter__(new_prompt_iterable)
        return

    def batch_add(
        self, new_prompt_iterables: List[Tuple[str, Generator[Prompt, None, None]]]
    ) -> None:
        """Store a collection of Prompt generators.

        Args:
            new_prompt_iterables (List[Tuple[str, Generator[Prompt, None, None]]]):
                List of tuples/lists containing:
                    - Molecular string and
                    - generator for Prompt objects.
        """
        new_db = [{k: v} for k, v in new_prompt_iterables]
        self.db += new_db
        return

    def get_iter(self, molecule: Molecule) -> Dict[str, Generator[Prompt, None, None]]:
        """Return Prompt generator for specific molecule.

        Args:
            molecule (Molecule): Molecular instance of interest.

        Returns:
            (Dict[str, Any]): Dictionary mapping between:
                - Molecule representation string
                - Prompt generator containing Prompt objects for different features.
        """
        representation = molecule.representation_string
        return list(filter(lambda x: x[0] == representation, self.db))[0]

    def _unravel(self, dictionary: Dict[str, Generator[Prompt, None, None]]) -> Dict[str, List[Dict[str, Any]]]:
        """Deconstruct inner generator objects.

        Args:
            dictionary (Dict[str, Generator[Prompt, None, None]]): Dictionary mapping between:
                - Molecule representation string
                - Prompt generator containing Prompt objects for different features.

        Returns:
            (Dict[str, List[Dict[str, Any]]]): Dictionary mapping between:
                - Molecule representation string
                - List of Prompt objects in Prompt generator for different features.
        """
        return {key: [v.__dict__() for v in value] for key, value in dictionary.items()}

    def __iter__(self):
        """Iterate over Prompt generator map.

        Args:
            (None)

        Yields:
            (Dict[str, List[Dict[str, Any]]]): Dictionary mapping between:
                - Molecule representation string
                - List of Prompt objects in Prompt generator for different features.
        """
        for d in self.db:
            yield self._unravel(d)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
