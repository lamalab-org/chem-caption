from chemcaption.featurize.stereochemistry import NumChiralCentersFeaturizer
from chemcaption.molecules import SMILESMolecule
import numpy as np


def test_num_chiral_centers():
    mol = SMILESMolecule(representation_string="CC")

    featurizer = NumChiralCentersFeaturizer()
    results = featurizer.featurize(mol)
    assert results == np.array([0.0])

    # taxol
    mol = SMILESMolecule(
        representation_string="CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C"
    )
    results = featurizer.featurize(mol)
    assert results == np.array([11.0])

    # now without stereoinfo in the SMILES
    mol = SMILESMolecule(
        representation_string="CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
    )

    results = featurizer.featurize(mol)
    assert results == np.array([11.0])
