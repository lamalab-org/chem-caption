# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.comparator submodule."""
import numpy as np

from chemcaption.featurize.comparator import (
    AtomCountComparator,
    DrugLikenessComparator,
    GhoseFilterComparator,
    IsoelectronicComparator,
    IsomerismComparator,
    IsomorphismComparator,
    LeadLikenessFilterComparator,
    LipinskiFilterComparator,
    ValenceElectronCountComparator,
)
from chemcaption.molecules import SMILESMolecule

# Implemented unit tests

__all__ = [
    "test_isoelectronic_comparator",
    "test_isomerism_comparator",
    "test_valence_electron_count_comparator",
    "test_isomorphism_comparator",
    "test_lipinski_violation_count_comparator",
    "test_ghose_filter_comparator",
    "test_lead_likeness_filter_comparator",
    "test_atom_count_comparator",
    "test_drug_likeness_comparator",
]


def test_isoelectronic_comparator():
    """Test for isoelectronic comparison."""
    isoelectronic_molecules = [
        SMILESMolecule("[C-]#[O+]"),  # Carbon II Oxide
        SMILESMolecule("N#N"),  # Nitrogen molecule
        SMILESMolecule("N#[O+]"),  # Nitrous Ion
        SMILESMolecule("[C-]#N"),  # Cyanide Ion
    ]

    featurizer = IsoelectronicComparator()

    results = featurizer.compare(isoelectronic_molecules).item()

    assert results == 1

    non_isoelectronic_molecules = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("[Mg+2]"),
        SMILESMolecule("[Cl-]"),
        SMILESMolecule("[C-]#[O+]"),
    ]

    results = featurizer.compare(non_isoelectronic_molecules).item()

    assert results == 0


def test_isomerism_comparator():
    """Test for isomericity."""
    isomeric_molecules = [
        SMILESMolecule("C1(Br)=CC=CC=C1Br"),  # 1,2-Dibromobenzene
        SMILESMolecule("C1=CC(=CC=C1Br)Br"),  # 1,4-Dibromobenzene
    ]

    featurizer = IsomerismComparator()

    results = featurizer.compare(isomeric_molecules).item()

    assert results == 1

    non_isomeric_molecules = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("[Mg+2]"),
        SMILESMolecule("[Cl-]"),
        SMILESMolecule("[C-]#[O+]"),
    ]

    results = featurizer.compare(non_isomeric_molecules).item()

    assert results == 0


def test_valence_electron_count_comparator():
    """Test for valence electron similarity."""
    valence_similar_electrons = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("[Mg+2]"),
    ]

    featurizer = ValenceElectronCountComparator()

    results = featurizer.compare(valence_similar_electrons).item()

    assert results == 1

    non_valence_similar_electrons = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("[Mg+2]"),
        SMILESMolecule("[Cl-]"),
        SMILESMolecule("[C-]#[O+]"),
    ]

    results = featurizer.compare(non_valence_similar_electrons).item()

    assert results == 0


def test_isomorphism_comparator():
    """Test for molecular structural similarity."""
    similar_structure_electrons = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("[Mg+2]"),
    ]

    featurizer = IsomorphismComparator()

    results = featurizer.compare(similar_structure_electrons).item()

    assert results == 1

    non_similar_structure_electrons = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("[Mg+2]"),
        SMILESMolecule("[Cl-]"),
        SMILESMolecule("[C-]#[O+]"),
    ]

    results = featurizer.compare(non_similar_structure_electrons).item()

    assert results == 0


def test_lipinski_violation_count_comparator():
    """Test for similarity via number of Lipinski rules violated."""
    lipinski_similar = [
        SMILESMolecule("C1(Br)=CC=CC=C1Br"),  # 1,2-Dibromobenzene
        SMILESMolecule("C1=CC(=CC=C1Br)Br"),  # 1,4-Dibromobenzene
    ]

    featurizer = LipinskiFilterComparator()

    results = featurizer.compare(lipinski_similar).item()

    assert results == 1

    non_lipinski_similar = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
    ]

    results = featurizer.compare(non_lipinski_similar).item()

    assert results == 0


def test_ghose_filter_comparator():
    """Test for similarity via number of Ghose filter violations."""
    ghose_similar = [
        SMILESMolecule("C1(Br)=CC=CC=C1Br"),  # 1,2-Dibromobenzene
        SMILESMolecule("C1=CC(=CC=C1Br)Br"),  # 1,4-Dibromobenzene
    ]

    featurizer = GhoseFilterComparator()

    results = featurizer.compare(ghose_similar).item()

    assert results == 1

    non_ghose_similar = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
    ]

    results = featurizer.compare(non_ghose_similar).item()

    assert results == 0


def test_lead_likeness_filter_comparator():
    """Test for similarity via number of Ghose filter violations."""
    lead_like = [
        SMILESMolecule("C1(Br)=CC=CC=C1Br"),  # 1,2-Dibromobenzene
        SMILESMolecule("C1=CC(=CC=C1Br)Br"),  # 1,4-Dibromobenzene
    ]

    featurizer = LeadLikenessFilterComparator()

    results = featurizer.compare(lead_like).item()

    assert results == 1

    non_lead_like = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
    ]

    results = featurizer.compare(non_lead_like).item()

    assert results == 0


def test_atom_count_comparator():
    """Test for similarity via number of atoms in molecule."""
    similar_atom_count = [
        SMILESMolecule("C1(Br)=CC=CC=C1Br"),  # 1,2-Dibromobenzene
        SMILESMolecule("C1=CC(=CC=C1Br)Br"),  # 1,4-Dibromobenzene
    ]

    featurizer = AtomCountComparator()

    results = featurizer.compare(similar_atom_count).item()

    assert results == 1

    non_similar_atom_count = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
    ]

    results = featurizer.compare(non_similar_atom_count).item()

    assert results == 0


def test_drug_likeness_comparator():
    """Test for similarity via molecular drug-likeness."""
    similar = [
        SMILESMolecule("C1(Br)=CC=CC=C1Br"),  # 1,2-Dibromobenzene
        SMILESMolecule("C1=CC(=CC=C1Br)Br"),  # 1,4-Dibromobenzene
    ]

    featurizer = DrugLikenessComparator()

    results = np.unique(featurizer.compare(similar))

    assert results == 1

    non_similar = [
        SMILESMolecule("[Na+]"),
        SMILESMolecule("CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
    ]

    results = np.unique(featurizer.compare(non_similar))

    assert results == 0
