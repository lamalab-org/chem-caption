from chemcaption.featurize.base import MultipleFeaturizer

from chemcaption.featurize.electronicity import HydrogenAcceptorCountFeaturizer
from chemcaption.featurize.stereochemistry import NumChiralCentersFeaturizer

from chemcaption.molecules import SMILESMolecule


def test_multiple_featurizer():
    smiles = SMILESMolecule("CCCC")

    featurizer = MultipleFeaturizer(
        featurizers=[
            HydrogenAcceptorCountFeaturizer(),
            NumChiralCentersFeaturizer(),
        ]
    )

    results = featurizer.featurize(smiles)
    assert len(results[0]) == 2
    assert len(results[0]) == len(featurizer.feature_labels())
