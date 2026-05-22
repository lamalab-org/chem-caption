# -*- coding: utf-8 -*-

"""Unit tests for `chemcaption.featurize.utils` submodule."""

import pytest

from chemcaption.featurize.utils import answer_generation

__all__ = [
    "test_answer_generation_basic",
    "test_answer_generation_wrong_input",
    "test_answer_generation_pluralization",
    "test_answer_generation_zero_amount",
    "test_answer_generation_single_element",
    "test_answer_generation_double_element",
    "test_answer_generation_three_elements",
    "test_answer_generation_mismatched_lengths_raises",
]


def test_answer_generation_basic():
    """Test answer_generation with multiple names and counts."""
    result = answer_generation([6, 12], ["carbon", "hydrogen"])
    assert result == "6 carbons and 12 hydrogens"

def test_answer_generation_wrong_input():
    """Test answer_generation with wrong input type."""
    with pytest.raises(ValueError, match="elements"):
        answer_generation(None, ["carbon", "hydrogen"])
    with pytest.raises(TypeError, match="elements"):
        answer_generation([6, None], ["carbon", "hydrogen"])
    with pytest.raises(TypeError, match="elements"):
        answer_generation([6, "string"], ["carbon", "hydrogen"])
    with pytest.raises(TypeError, match="elements"):
        answer_generation([None, 42], [6, 12])
    with pytest.raises(TypeError, match="elements"):
        answer_generation(["string", 42], [6, 12])

def test_answer_generation_pluralization():
    """Test that amount == 1 uses singular form without trailing 's'."""
    result = answer_generation([1], ["nitrogen"])
    assert result == "1 nitrogen"

def test_answer_generation_zero_amount():
    """Test that amount == 0 uses 'no <name>s' phrasing."""
    result = answer_generation([0], ["sulfur"])
    assert result == "no sulfurs"

def test_answer_generation_single_element():
    """Test answer_generation returns unprefixed string for a single name/count pair."""
    result = answer_generation([2], ["oxygen"])
    assert result == "2 oxygens"

def test_answer_generation_double_element():
    """Test answer_generation returns unprefixed string for a single name/count pair."""
    result = answer_generation([2, 1], ["hydrogen", "oxygen"])
    assert result == "2 hydrogens and 1 oxygen"

def test_answer_generation_three_elements():
    """Test answer_generation with three elements uses comma-and style."""
    result = answer_generation([6, 12, 1], ["carbon", "hydrogen", "oxygen"])
    assert result == "6 carbons, 12 hydrogens, and 1 oxygen"

def test_answer_generation_mismatched_lengths_raises():
    """Test that mismatched names and elements lengths raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        answer_generation([6, 12], ["carbon"])
    with pytest.raises(ValueError, match="Length mismatch"):
        answer_generation([6], ["carbon", "hydrogen"])