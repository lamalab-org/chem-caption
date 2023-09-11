# -*- coding: utf-8 -*-

"""Classes for representing featurizer output as text."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

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
    prompt_template: Optional[str] = None
    completion_template: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of object.

        Args:
            None.

        Returns:
            (dict): Dictionary containing all relevant prompt-related information.
        """

        return self.__dict__()
    

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
            "prompt_template": self.prompt_template,
            "completion_template": self.completion_template,
            "completion": self.completion,
            "completion_names": self.completion_names,
            "completion_labels": self.completion_labels,
            "filled_prompt": self.fill_template(self.prompt_template),
            "filled_completion": self.fill_template(self.completion_template),
        }

    def fill_template(self, template, precision_type: str = "decimal") -> str:
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
            COMPLETION=self.completion,
        )
        molecular_info = inspect_info(molecular_info)

        return template.format(**molecular_info)

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
