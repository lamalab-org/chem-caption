# -*- coding: utf-8 -*-

"""Unit tests for `chemcaption.featurize.utils` submodule."""

import pytest

from chemcaption.featurize.utils import answer_generation, join_list_elements

__all__ = [
    "test_answer_generation_basic",
    "test_answer_generation_wrong_input",
    "test_answer_generation_single_element",
    "test_answer_generation_pluralization",
    "test_answer_generation_zero_amount",
    "test_answer_generation_three_elements",
    "test_answer_generation_mismatched_lengths_raises",
    "test_join_list_elements_single",
    "test_join_list_elements_multiple",
]


def test_answer_generation_basic():
    """Test answer_generation with multiple names and counts."""
    result = answer_generation(["carbon", "hydrogen"], [6, 12])
    assert result == "6 carbons and 12 hydrogens"
    
def test_answer_generation_wrong_input():
    """Test answer_generation with wrong input type."""
    with pytest.raises(ValueError, match="elements"):
        answer_generation(["carbon", "hydrogen"], None)
    with pytest.raises(ValueError, match="names"):
        answer_generation(None, ["carbon", "hydrogen"])
    with pytest.raises(TypeError, match="elements"):
        answer_generation(["carbon", "hydrogen"], [6, None])
    with pytest.raises(TypeError, match="elements"):
        answer_generation(["carbon", "hydrogen"], [6, "string"])
    with pytest.raises(TypeError, match="names"):
        answer_generation([None, 42], [6, 12])
    with pytest.raises(TypeError, match="names"):
        answer_generation(["string", 42], [6, 12])


def test_answer_generation_single_element():
    """Test answer_generation returns unprefixed string for a single name/count pair."""
    result = answer_generation(["oxygen"], [2])
    assert result == "2 oxygens"


def test_answer_generation_pluralization():
    """Test that amount == 1 uses singular form without trailing 's'."""
    result = answer_generation(["nitrogen"], [1])
    assert result == "1 nitrogen"


def test_answer_generation_zero_amount():
    """Test that amount == 0 uses 'no <name>s' phrasing."""
    result = answer_generation(["sulfur"], [0])
    assert result == "no sulfurs"


def test_answer_generation_three_elements():
    """Test answer_generation with three elements uses comma-and style."""
    result = answer_generation(["carbon", "hydrogen", "oxygen"], [6, 12, 1])
    assert result == "6 carbons, 12 hydrogens, and 1 oxygen"


def test_answer_generation_mismatched_lengths_raises():
    """Test that mismatched names and elements lengths raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        answer_generation(["carbon", "hydrogen"], [6])

def test_join_list_elements_single():
    """Test join_list_elements with a single element."""
    assert join_list_elements([7]) == "7"


def test_join_list_elements_multiple():
    """Test join_list_elements with multiple elements uses comma-and style."""
    assert join_list_elements([1, 2, 3]) == "1, 2, and 3"
