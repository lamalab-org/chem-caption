# -*- coding: utf-8 -*-

"""Test feature names and labels for uniqueness."""

from typing import Dict, List, Tuple, Union

from chemcaption.export.export import FEATURIZER
from chemcaption.featurize.base import MultipleFeaturizer

__all__ = [
    "get_repetitive_labels",
]


def get_repetitive_labels(
    featurizer: MultipleFeaturizer,
) -> Tuple[Dict[str, Dict[str, Union[int, List[str]]]], List[str]]:
    """Returns all repeated labels.

    Args:
        featurizer (MultipleFeaturizer): MultipleFeaturizer instance.

    Returns:
        Tuple[Dict[str, Dict[str, Union[int, List[str]]]], List[str]]: Tuple containing:
            - Dictionary of information for repeated labels and
            - List of all labels.
    """
    all_labels = featurizer.feature_labels()
    repetitive_labels = {}

    for f in featurizer.featurizers:
        labels = f.feature_labels()
        for label in labels:
            if label in all_labels:
                if label not in repetitive_labels:
                    info = {"count": 1, "appearance": [f.__class__.__name__]}
                    repetitive_labels[label] = info
                else:
                    repetitive_labels[label]["count"] += 1
                    repetitive_labels[label]["appearance"].append(f.__class__.__name__)

    return {
        key: value for key, value in repetitive_labels.items() if value["count"] > 1
    }, all_labels


if __name__ == "__main__":
    repetitive_labels, all_labels = get_repetitive_labels(FEATURIZER)

    print("Diagnostics:")
    print("=" * 50)
    print("Number of featurizers implemented:", len(FEATURIZER.featurizers))
    print("=" * 50)
    print("Number of labels:", len(all_labels))
    print("Number of repeated labels:", len(repetitive_labels))
    print("Number of unique labels:", len(set(all_labels)))

    if len(repetitive_labels) == 0:
        print("=" * 50)
        print("All detected labels:\n")

        for i, label in enumerate(all_labels, 1):
            print(f"{i:.3g}.", label)
        exit()

    for k, v in repetitive_labels.items():
        if v["count"] > 1:
            print(f"{k}:", v)

    raise Exception(
        f"Number of redundant labels found: {len(repetitive_labels)}. Please fix as soon as possible!"
    )
