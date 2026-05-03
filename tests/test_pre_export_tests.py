# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.export.pre_export_tests submodule."""

from chemcaption.export.pre_export_tests import get_repetitive_labels, get_smarts_featurizers
from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.featurize.bonds import BondTypeCountFeaturizer, BondTypeProportionFeaturizer
from chemcaption.featurize.substructure import FragmentSearchFeaturizer

from chemcaption.featurize.bonds import BondTypeCountFeaturizer, BondTypeProportionFeaturizer

from chemcaption.featurize.base import MultipleFeaturizer

__all__ = ["test_pre_export_test"]


def test_pre_export_test():
    """Tests the pre_export_tests"""
    smarts = get_smarts_featurizers()

    assert isinstance(smarts, list)
    assert isinstance(smarts[0], FragmentSearchFeaturizer)

    featurizer = MultipleFeaturizer([BondTypeCountFeaturizer(), BondTypeProportionFeaturizer()])

    repetitive, _ = get_repetitive_labels(featurizer)

    assert isinstance(repetitive, dict)
    assert not repetitive
