# -*- coding: utf-8 -*-

"""Test chemcaption.featurize.text classes."""

import pytest

from chemcaption.featurize.text import Prompt
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, generate_prompt_test_data

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Element mass-related presets
PRESET = ["carbon", "hydrogen", "oxygen", "nitrogen", "phosphorus"]

# Implemented tests for text-related classes.

__all__ = [
    "test_prompt",
]


"""Test for Prompt object capabilities."""


# @pytest.mark.parametrize(
#     "test_input, template, expected",
#     generate_prompt_test_data(
#         property_bank=PROPERTY_BANK,
#         representation_name=KIND.upper(),
#         property=list(map(lambda x: "num_" + x + "_atoms", PRESET)),
#         key="multiple",
#     ),
# )
# def test_prompt(test_input, template, expected):
#     """Test Prompt object for prompt template formatting."""
#     prompt = Prompt(
#         template=template,
#         completion=test_input["PROPERTY_VALUE"],
#         completion_names=test_input["PROPERTY_NAME"],
#         completion_type=(
#             [type(t) for t in test_input["PROPERTY_VALUE"]]
#             if isinstance(test_input["PROPERTY_VALUE"], list)
#             else type(test_input["PROPERTY_VALUE"])
#         ),
#         representation=test_input["REPR_STRING"],
#         representation_type=test_input["REPR_SYSTEM"],
#     )
#     result = prompt.__str__()
#     assert result == expected
