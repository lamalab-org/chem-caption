# -*- coding: utf-8 -*-

"""Test feature names for uniqueness."""

from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.export.export import FEATURIZER


def get_repetitive_labels(featurizer: MultipleFeaturizer):

    all_labels = []
    repetitive_labels = {}

    for f in featurizer.featurizers:
        labels = f.feature_labels()
        for label in labels:
            all_labels.append(label)
            if label not in repetitive_labels:
                info = {
                    "count": 1,
                    "appearance": [f.__class__.__name__]
                }
                repetitive_labels[label] = info
            else:
                repetitive_labels[label]["count"] += 1
                repetitive_labels[label]["appearance"].append(f.__class__.__name__)

    return repetitive_labels, all_labels


if __name__ == "__main__":
    repetitive_labels, all_labels = get_repetitive_labels(FEATURIZER)

    print(repetitive_labels)
    print(all_labels)

    print("Number of labels:", len(all_labels))
    print("Number of repeated labels:", len(repetitive_labels))
    print("Number of unique labels:", len(set(all_labels)))

    for k, v in repetitive_labels.items():
        if v["count"] > 1:
            print(f"{k}", v)
