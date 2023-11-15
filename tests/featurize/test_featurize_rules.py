from chemcaption.featurize.rules import LipinskiViolationCountFeaturizer
from chemcaption.molecules import SMILESMolecule


def test_lipinski_violation_count_featurizer():
    molecule = SMILESMolecule("O")
    featurizer = LipinskiViolationCountFeaturizer()
    results = featurizer.featurize(molecule)
    assert results[0] == 0

    molecule = SMILESMolecule("CCCCCCCCCCCCCCCC")
    featurizer = LipinskiViolationCountFeaturizer()
    results = featurizer.featurize(molecule)
    assert results[0] == 1

    text = featurizer.text_featurize(pos_key="noun", molecule=molecule)
    assert (
        text.to_dict()["filled_prompt"]
        == "Question: What is the number of Lipinski violations of the molecule with SMILES CCCCCCCCCCCCCCCC?"
    )
    assert text.to_dict()["filled_completion"] == "Answer: 1"
