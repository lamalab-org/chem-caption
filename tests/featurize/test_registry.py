# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.registry submodule."""

import chemcaption

from chemcaption.featurize.base import AbstractFeaturizer, AbstractComparator

from chemcaption.featurize.registry import init_all_comparators, init_all_featurizers

__all__ = [
    "test_registry",
]


def test_registry():
    """Tests the registry functionality."""

    featurizers = init_all_featurizers(chemcaption.featurize.bonds)

    assert isinstance(featurizers, list)
    assert [isinstance(f, AbstractFeaturizer) for f in featurizers]

    comparators = init_all_comparators(chemcaption.featurize.comparator)

    assert isinstance(comparators, list)
    assert [isinstance(c, AbstractComparator) for c in comparators]
