# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.rules submodule."""

import numpy as np

from chemcaption.featurize.reaction import (
    SolventAccessibleAtomAreaFeaturizer,
    SolventAccessibleSurfaceAreaFeaturizer,
    SolventAccessibleVolumeFeaturizer,
)
from chemcaption.molecules import SMILESMolecule

__all__ = [
    "test_solvent_accessible_surface_area_featurizer",
    "test_solvent_accessible_volume_featurizer",
    "test_solvent_accessible_atom_area_featurizer",
]


def test_solvent_accessible_surface_area_featurizer():
    """Tests featurizer SolventAccessibleSurfaceAreaFeaturizer."""

    molecule = SMILESMolecule("O")
    featurizer = SolventAccessibleSurfaceAreaFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    results = featurizer.featurize(molecule)
    assert len(results) > 0
    assert len(results[0]) == len(featurizer.feature_labels)

    featurizer = SolventAccessibleSurfaceAreaFeaturizer(qc_optimize=True)
    results = featurizer.featurize(molecule)
    assert len(results) > 0
    assert len(results[0]) == len(featurizer.feature_labels)


def test_solvent_accessible_volume_featurizer():
    """Tests featurizer SolventAccessibleVolumeFeaturizer."""

    molecule = SMILESMolecule("O")
    featurizer = SolventAccessibleVolumeFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    results = featurizer.featurize(molecule)
    assert len(results) > 0
    assert len(results[0]) == len(featurizer.feature_labels)

    featurizer = SolventAccessibleVolumeFeaturizer(qc_optimize=True)
    results = featurizer.featurize(molecule)
    assert len(results) > 0
    assert len(results[0]) == len(featurizer.feature_labels)


def test_solvent_accessible_atom_area_featurizer():
    """Tests featurizer SolventAccessibleAtomAreaFeaturizer."""

    molecule = SMILESMolecule("O")
    featurizer = SolventAccessibleAtomAreaFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    results = featurizer.featurize(molecule)
    assert len(results) > 0
    assert len(results[0]) == len(featurizer.feature_labels)

    mean_res = np.mean(results[results > 8])

    featurizer = SolventAccessibleAtomAreaFeaturizer(qc_optimize=True)
    results = featurizer.featurize(molecule)
    assert len(results) > 0
    assert len(results[0]) == len(featurizer.feature_labels)

    featurizer.max_index = None
    results = featurizer.featurize(molecule)
    assert len(results) > 0
    assert len(results[0]) == len(featurizer.feature_labels)

    featurizer = SolventAccessibleAtomAreaFeaturizer(aggregation="mean")

    r = featurizer.featurize(molecule)
    assert len(r) > 0
    assert len(r[0]) == len(featurizer.feature_labels)
    assert r[0][0] == mean_res

    featurizer = SolventAccessibleAtomAreaFeaturizer()
    mols = [SMILESMolecule("O"), SMILESMolecule("C")]
    results = featurizer.featurize_many(mols)

    assert len(results) == len(mols)
