# -*- coding: utf-8 -*-

"""Implementations for utlity functions to generate all comparators in a submodule."""

from abc import ABCMeta

import chemcaption
from chemcaption.featurize.base import (
    AbstractComparator,
    AbstractFeaturizer,
    Comparator,
    MultipleComparator,
    MultipleFeaturizer,
)


def init_all_featurizers(module) -> list:
    """Returns a list of initialized featurizers per chemcaption submodule."""

    classes = []
    for item in module.__dict__.values():
        if isinstance(item, ABCMeta):
            if not issubclass(item, AbstractFeaturizer):
                continue

            try:
                f = item()
            except Exception:
                continue

            if isinstance(f, MultipleFeaturizer):
                continue

            classes.append(f)

    return classes


def init_all_comparators(module) -> list:
    """Returns a list of initialized comparators per chemcaption submodule."""

    classes = []
    for item in module.__dict__.values():
        if isinstance(item, ABCMeta):
            if not issubclass(item, AbstractComparator):
                continue

            try:
                f = item()
            except Exception:
                continue

            if isinstance(f, MultipleComparator):
                continue

            if type(f) == Comparator:
                continue

            classes.append(f)

    return classes


"""Featurizers"""

BONDS_FEATURIZERS = init_all_featurizers(chemcaption.featurize.bonds)
COMPOSITION_FEATURIZERS = init_all_featurizers(chemcaption.featurize.composition)
ELECTRONICITY_FEATURIZERS = init_all_featurizers(chemcaption.featurize.electronicity)
MISCELLANEOUS_FEATURIZERS = init_all_featurizers(chemcaption.featurize.miscellaneous)
REACTION_FEATURIZERS = init_all_featurizers(chemcaption.featurize.reaction)
RULES_FEATURIZERS = init_all_featurizers(chemcaption.featurize.rules)
SPATIAL_FEATURIZERS = init_all_featurizers(chemcaption.featurize.spatial)
STEREOCHEMISTRY_FEATURIZERS = init_all_featurizers(chemcaption.featurize.stereochemistry)
SUBSTRUCTURE_FEATURIZERS = init_all_featurizers(chemcaption.featurize.substructure)
SYMMETRY_FEATURIZERS = init_all_featurizers(chemcaption.featurize.symmetry)

"""Comparators"""

COMPARATORS = init_all_comparators(chemcaption.featurize.comparator)
