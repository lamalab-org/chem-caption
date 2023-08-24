# -*- coding: utf-8 -*-

"""Classes for representing featurizer output as text."""

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import numpy as np

from chemcaption.featurize.text_utils import inspect_info

# Implemented text-related classes

__all__ = [
    "Prompt",
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
            "filled_prompt": self.fill_template(),
        }

    def fill_template(self, precision_type: str = "decimal") -> str:
        """Fill up the prompt template with appropriate values.

        Args:
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

        return self.template.format(**molecular_info)

    def __str__(self) -> str:
        """Return string representation of object.

        Args:
            None.

        Returns:
            (str): Appropriately formatted template.
        """
        return self.fill_template()

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
    def __init__(self, prompt_iterable: Optional[Union[str, Generator[Prompt, None, None]]] = None):
        self.db = [{prompt_iterable[0]: prompt_iterable[1]}] if prompt_iterable is not None else []

    def __add_iter__(
        self, new_prompt_iterable: Optional[Union[str, Generator[Prompt, None, None]]]
    ):
        self.db.append({new_prompt_iterable[0]: new_prompt_iterable[1]})
        return

    def add(self, new_prompt_iterable: Optional[Union[str, Generator[Prompt, None, None]]]):
        self.__add_iter__(new_prompt_iterable)
        return

    def batch_add(self, new_prompt_iterables: List[Union[str, Generator[Prompt, None, None]]]):
        new_db = [{k: v} for k, v in new_prompt_iterables]
        self.db += new_db
        return

    def get_iter(self, representation: str) -> Dict[str, Any]:
        return list(filter(lambda x: x[0] == representation, self.db))[0]

    def _unravel(self, dictionary):
        return {key: [v.__dict__() for v in value] for key, value in dictionary.items()}

    def __iter__(self):
        for d in self.db:
            yield self._unravel(d)
