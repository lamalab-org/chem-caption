# -*- coding: utf-8 -*-

"""Utilities for facilitating text featurization."""

from random import shuffle
from typing import Dict, List, Union

import numpy as np

# Implemented text-related utilities

__all__ = [
    "TEXT_TEMPLATES",  # Constant
    "QA_TEMPLATES",  # Constant
    "generate_template",  # Utility function
    "inspect_template",  # Utility function
    "inspect_info",  # Utility function
    "generate_info",  # Utility function
]

"""Prompt templates"""

TEXT_TEMPLATES = dict(
    single=[
        "The {PROPERTY_NAME} property has a magnitude {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} property has a magnitude of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} has a magnitude {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} has a magnitude of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} property has a value of {PROPERTY_VALUE}.",
        "The value of {PROPERTY_NAME} property is {PROPERTY_VALUE}.",
        "The value for {PROPERTY_NAME} property is {PROPERTY_VALUE}.",
        "The value of {PROPERTY_NAME} is {PROPERTY_VALUE}.",
        "The value for {PROPERTY_NAME} is {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} has a value of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} has a value {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} has a magnitude of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} has a magnitude {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} property has a value of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} property has a value {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} property has a magnitude of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} property has a magnitude {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} property is evaluated to be {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} property is evaluated to have a value of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} is evaluated to be {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} property is evaluated to have a magnitude of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} is evaluated to have a value of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} is evaluated to have a magnitude of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} is measured to have a value of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} is measured to have a magnitude of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} for the molecule with representation `{REPR_STRING}` in the {REPR_SYSTEM} representational"
        " system has a value of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} for the molecule with representation `{REPR_STRING}` in the {REPR_SYSTEM} representational"
        " system is {PROPERTY_VALUE}.",
        "The molecule represented by representation `{REPR_STRING}` via the {REPR_SYSTEM} representational system "
        "is characterized by the {PROPERTY_NAME} property as {PROPERTY_VALUE}.",
    ],
    multiple=[
        "The {PROPERTY_NAME} properties have the respective magnitudes: {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have the respective magnitudes of: {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have the magnitudes {PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} properties have respective magnitudes {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have respective magnitudes of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have magnitudes {PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} properties have respective magnitudes {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have the following magnitudes {PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} properties have the following magnitudes respectively: {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have the following respective magnitudes {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have the respective values: {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have the respective values of: {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have the values {PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} properties have respective values {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have respective values of {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have values {PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} properties have respective values {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have the following values {PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} properties have the following values respectively: {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have the following respective values {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have been evaluated to be {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have been evaluated to have values of {PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} properties have been evaluated to have magnitudes of {PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} properties have been evaluated to have the following values "
        "{PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} properties have been evaluated to have following magnitudes {PROPERTY_VALUE} "
        "respectively.",
        "The {PROPERTY_NAME} properties have been evaluated to have the following respective values {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} properties have been evaluated to have following respective magnitudes {PROPERTY_VALUE}.",
        "The {PROPERTY_NAME} for the molecule with representation `{REPR_STRING}` in the {REPR_SYSTEM} "
        "representational system have the values: {PROPERTY_VALUE} respectively.",
        "The {PROPERTY_NAME} for the molecule with representation `{REPR_STRING}` in the {REPR_SYSTEM} "
        "representational system are {PROPERTY_VALUE} respectively.",
        "The values for the {PROPERTY_NAME} of the molecule with representation `{REPR_STRING}` in the "
        "{REPR_SYSTEM} representational system are {PROPERTY_VALUE}.",
        "The magnitudes for the {PROPERTY_NAME} of the molecule with representation `{REPR_STRING}` in the "
        "{REPR_SYSTEM} representational system are {PROPERTY_VALUE}.",
        "The molecule represented by representation `{REPR_STRING}` via the {REPR_SYSTEM} representational "
        "system is characterized by the following properties, having the respective values: "
        "{PROPERTY_NAME}, and {PROPERTY_VALUE}.",
    ],
)

QA_TEMPLATES = dict(
    multiple=[
        "What are the values of the {PROPERTY_NAME} properties of {REPR_SYSTEM} molecule `{REPR_STRING}`?",
        "What are the values of {PROPERTY_NAME} properties of {REPR_SYSTEM} molecule `{REPR_STRING}`?",
        "For the {REPR_SYSTEM} molecule with string `{REPR_STRING}`, what are the values of the "
        "{PROPERTY_NAME} properties?",
        "For the {REPR_SYSTEM} molecule with string `{REPR_STRING}`, what are the values of {PROPERTY_NAME}?",
        "For the {REPR_SYSTEM} molecule with string `{REPR_STRING}`, what are the {PROPERTY_NAME} values?",
        "For the {REPR_SYSTEM} molecule with string `{REPR_STRING}`, what are the {PROPERTY_NAME}?",
        "What values do the {REPR_SYSTEM} molecule `{REPR_STRING}` have for the {PROPERTY_NAME}?",
        "What are the values for the following properties of {REPR_SYSTEM} molecule with {REPR_SYSTEM} string"
        " `{REPR_STRING}`: {PROPERTY_NAME}?",
        "What values do the {REPR_SYSTEM} molecule `{REPR_STRING}` have for the properties: {PROPERTY_NAME}?",
        "What values do the {REPR_SYSTEM} molecule `{REPR_STRING}` have for the {PROPERTY_NAME} properties?",
    ],
    single=[
        "What is the value of the {PROPERTY_NAME} property for {REPR_SYSTEM} molecule `{REPR_STRING}`?",
        "What is the value of {PROPERTY_NAME} property of {REPR_SYSTEM} molecule `{REPR_STRING}`?",
        "What value does {REPR_SYSTEM} molecule `{REPR_STRING}` have for the {PROPERTY_NAME} property?",
        "What value does the {REPR_SYSTEM} molecule `{REPR_STRING}` have for the {PROPERTY_NAME} property?",
        "What value does {REPR_SYSTEM} molecule `{REPR_STRING}` have for the {PROPERTY_NAME}?",
        "What value does the {REPR_SYSTEM} molecule `{REPR_STRING}` have for the {PROPERTY_NAME}?",
        "For the {REPR_SYSTEM} molecule with string `{REPR_STRING}`,"
        " what is the value of the {PROPERTY_NAME} property?",
        "For the {REPR_SYSTEM} molecule with string `{REPR_STRING}`, what is the value of {PROPERTY_NAME}?",
        "For the {REPR_SYSTEM} molecule with string `{REPR_STRING}`, what is the {PROPERTY_NAME} value?",
        "For the {REPR_SYSTEM} molecule with string `{REPR_STRING}`, what is the {PROPERTY_NAME}?",
        "What is the magnitude of the {PROPERTY_NAME} property for {REPR_SYSTEM} molecule `{REPR_STRING}`?",
        "What magnitude does the {REPR_SYSTEM} molecule `{REPR_STRING}` have for the property: {PROPERTY_NAME}?",
        "What magnitude does the {REPR_SYSTEM} molecule `{REPR_STRING}` have for the {PROPERTY_NAME} property?",
        "What magnitude does the {REPR_SYSTEM} molecule `{REPR_STRING}` have for the {PROPERTY_NAME}?",
    ],
)


"""Utility functions."""


def generate_template(template_type: str = "qa", key: str = "single") -> str:
    """Randomly select prompt template.

    Args:
        template_type (str): Type of template. Take either `qa` or `text`. Defaults to `qa`.
        key (str): Cardinality of template. Can be `single` or `multiple`. Defaults to `single`.

    Returns:
        str: Selected template.
    """
    templates = QA_TEMPLATES[key] if template_type == "qa" else TEXT_TEMPLATES[key]

    template = templates[np.random.randint(low=0, high=len(templates), size=(1,)).item()]
    return template


def inspect_info(info: dict) -> Dict[str, Union[str, List[int], List[float]]]:
    """Inspect information dictionary and update contents if necessary.

    Args:
        info (dict): Dictionary of molecular information.

    Returns:
        Dict[str, Union[str, List[int], List[float]]]: Updated dictionary of molecular information.

    """
    new_info = info.copy()
    for key, value in new_info.items():
        # Process each item in the dictionary
        if key == "PRECISION":
            continue

        elif isinstance(value, (list, tuple)):
            list_len = len(value)
            value = [
                (
                    str(round(sub_value, new_info["PRECISION"]))
                    if isinstance(sub_value, (int, float))
                    else str(sub_value)
                )
                for sub_value in value
            ]

            if list_len > 2:
                properties = ", ".join(value[:-1])
                properties += ", and " + value[-1]
            elif list_len == 2:
                properties = " and ".join(value)
            else:
                properties = value[0]
        else:
            if key == "PRECISION_TYPE":
                properties = "decimal places" if value == "decimal" else "significant figures"

            else:
                properties = (
                    str(round(value, new_info["PRECISION"]))
                    if isinstance(value, (int, float))
                    else str(value)
                )

        # Store processed information in new dictionary
        new_info[key] = properties
    return new_info


def inspect_template(template: str, template_cardinality: str = "single") -> str:
    """Inspect and mutate template structure on the fly.

    Args:
        template (str): Template format as string.
        template_cardinality (str): Type of template. May be `multiple` or `single`. Defaults to `single`.

    Returns:
        str: Updated template.
    """
    prob = np.random.randn()

    if prob > 0.5:
        pass
    else:
        if template_cardinality == "single":
            hot_words = [
                "{PROPERTY_NAME} value",
                "{PROPERTY_NAME}",
                "magnitude",
                "value",
            ]

        else:
            hot_words = [
                "{PROPERTY_NAME} values",
                "{PROPERTY_NAME}",
                "magnitudes",
                "values",
            ]

        shuffle(hot_words)

        for term in hot_words:
            if term in template:
                prob = np.random.randn()
                if prob > 0.5:
                    template = template.split(term, maxsplit=1)
                    template = (
                        template[0]
                        + term
                        + " (rounded to within {PRECISION} {PRECISION_TYPE})"
                        + template[-1]
                    )
                    break
                else:
                    pass
            else:
                continue

    return template


def generate_info(info_cardinality: str = "single") -> Dict[str, Union[str, float, int]]:
    """Generate dictionary of molecular information at random.

    Args:
        info_cardinality (str): Cardinality of template. Can be `single` or `multiple`. Defaults to `single`.

    Returns:
        Dict[str, Union[str, float, int]]: Dictionary of molecular information.

    """
    if info_cardinality == "single":
        info = dict(
            PROPERTY_NAME="molar mass",
            REPR_SYSTEM="SMILES",
            REPR_STRING="CCC",
            PROPERTY_VALUE=359.02,
            PRECISION=2,
            PRECISION_TYPE="decimal",
        )
    else:
        info = dict(
            PROPERTY_NAME=["valence", "density", "molar mass"],
            REPR_SYSTEM="SMILES",
            REPR_STRING="CCCC(C)C",
            PROPERTY_VALUE=[27, 0.37, 359.02],
            PRECISION=2,
            PRECISION_TYPE="decimal",
        )

    return info
