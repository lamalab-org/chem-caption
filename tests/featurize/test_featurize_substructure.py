from chemcaption.featurize.substructure import TopologyCountFeaturizer, SMARTSFeaturizer
from chemcaption.molecules import SMILESMolecule
from chemcaption.featurize.text import Prompt


def test_topology_count_featurizer():
    molecule = SMILESMolecule("C1=CC=CC=C1")
    featurizer = TopologyCountFeaturizer.from_presets("carbon")
    results = featurizer.featurize(molecule)
    assert len(results) == 1
    assert len(results[0]) == len(featurizer.feature_labels())
    text = featurizer.text_featurize(molecule)
    assert isinstance(text, Prompt)
