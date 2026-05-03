# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.electronicity submodule."""

import numpy as np

from chemcaption.featurize.electronicity import (
    AtomChargeFeaturizer,
    AtomElectrophilicityFeaturizer,
    AtomNucleophilicityFeaturizer,
    ElectronAffinityFeaturizer,
    HOMOEnergyFeaturizer,
    HydrogenAcceptorCountFeaturizer,
    HydrogenDonorCountFeaturizer,
    IonizationPotentialFeaturizer,
    LUMOEnergyFeaturizer,
    MoleculeElectrofugalityFeaturizer,
    MoleculeElectrophilicityFeaturizer,
    MoleculeNucleofugalityFeaturizer,
    MoleculeNucleophilicityFeaturizer,
    ValenceElectronCountFeaturizer,
)
from chemcaption.molecules import SMILESMolecule

__all__ = [
    "test_hydrogen_acceptor_count_featurizer",
    "test_hydrogen_donor_count_featurizer",
    "test_valence_electron_count_featurizer",
    "test_electron_affinity_featurizer",
    "test_ionization_potential_featurizer",
    "test_homo_energy_featurizer",
    "test_lumo_energy_featurizer",
    "test_atom_nucleophilicity_featurizer",
    "test_atom_electrophilicity_featurizer",
    "test_molecule_nucleophilicity_featurizer",
]


def test_molecule_electrofugality_featurizer():
    """Test the MoleculeElectrofugalityFeaturizer."""
    featurizer = MoleculeElectrofugalityFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_molecule_nucleofugality_featurizer():
    """Test the MoleculeNucleofugalityFeaturizer."""
    featurizer = MoleculeNucleofugalityFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_molecule_electrophilicity_featurizer():
    """Test the MoleculeElectrophilicityFeaturizer."""
    featurizer = MoleculeElectrophilicityFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_molecule_nucleophilicity_featurizer():
    """Test the MoleculeNucleophilicityFeaturizer."""
    featurizer = MoleculeNucleophilicityFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_atom_electrophilicity_featurizer():
    """Test the AtomElectrophilicityFeaturizer."""
    featurizer = AtomElectrophilicityFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_atom_nucleophilicity_featurizer():
    """Test the AtomNucleophilicityFeaturizer."""
    featurizer = AtomNucleophilicityFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_atom_charge_featurizer():
    """Test the AtomChargeFeaturizer."""
    featurizer = AtomChargeFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_lumo_energy_featurizer():
    """Test the LUMOEnergyFeaturizer."""
    featurizer = LUMOEnergyFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_homo_energy_featurizer():
    """Test the HOMOEnergyFeaturizer."""
    featurizer = HOMOEnergyFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_ionization_potential_featurizer():
    """Test the IonizationPotentialFeaturizer."""
    featurizer = IonizationPotentialFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])


def test_hydrogen_acceptor_count_featurizer():
    """Test HydrogenAcceptorCountFeaturizer."""
    featurizer = HydrogenAcceptorCountFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.equal(results, 4).all()
    assert len(featurizer.feature_labels) == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert text.to_dict()["filled_prompt"] == (
        "Question: What is the number of hydrogen bond acceptors of the "
        "molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 4"


def test_hydrogen_donor_count_featurizer():
    """Test HydrogenDonorCountFeaturizer."""
    featurizer = HydrogenDonorCountFeaturizer()
    assert isinstance(featurizer.implementors(), list)
    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.equal(results, 1).all()
    assert len(featurizer.feature_labels) == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of hydrogen bond donors of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 1"


def test_valence_electron_count_featurizer():
    """Test ValenceElectronCountFeaturizer."""
    featurizer = ValenceElectronCountFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert np.equal(results, 56).all()
    assert len(featurizer.feature_labels) == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of valence electrons of the molecule with SMILES O=C1C=CC(=O)C(C(=O)O)=C1?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 56"


def test_electron_affinity_featurizer():
    """Test the ElectronAffinityFeaturizer."""
    featurizer = ElectronAffinityFeaturizer()
    assert isinstance(featurizer.implementors(), list)

    molecule = SMILESMolecule("O=C1C=CC(=O)C=C1C(=O)O")

    results = featurizer.featurize(molecule)

    assert len(featurizer.feature_labels) == len(results[0])
