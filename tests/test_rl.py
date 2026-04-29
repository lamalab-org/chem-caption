import numpy as np
import pytest

from chemcaption.featurize.composition import AtomCountFeaturizer, MolecularFormulaFeaturizer
from chemcaption.molecules import SMILESMolecule
from chemcaption.rl import ChemCaptionEnv, FeatureReward, FeatureTarget


def test_env_exposes_raw_featurizer_target():
    featurizer = MolecularFormulaFeaturizer()
    env = ChemCaptionEnv(
        molecules=["O"],
        featurizer=featurizer,
        include_target_in_observation=True,
    )

    observation, info = env.reset(options={"index": 0})

    assert observation["representation_string"] == "O"
    assert "raw_featurizer_output" in observation
    assert np.array_equal(
        info["raw_featurizer_output"],
        featurizer.featurize(SMILESMolecule("O")),
    )
    assert info["target_labels"] == ["molecular_formula"]
    assert info["target"].tolist() == ["H2O"]
    assert info["target_labeled"] == {"molecular_formula": "H2O"}
    assert env.raw_featurizer_output is info["raw_featurizer_output"]


def test_env_step_rewards_array_actions():
    featurizer = AtomCountFeaturizer()
    env = ChemCaptionEnv(molecules=["O"], featurizer=featurizer)
    observation, info = env.reset()

    assert "target" not in observation

    next_observation, reward, terminated, truncated, step_info = env.step(info["target"])

    assert next_observation["representation_string"] == "O"
    assert reward == 1.0
    assert terminated
    assert not truncated
    assert step_info["num_matches"] == 1
    assert step_info["reward_metric"] == "exact"


def test_env_step_rewards_mapping_actions():
    env = ChemCaptionEnv(
        molecules=[SMILESMolecule("O")],
        featurizer=AtomCountFeaturizer(),
    )
    _, info = env.reset()

    _, reward, _, _, step_info = env.step({"num_atoms": info["target_labeled"]["num_atoms"]})

    assert reward == 1.0
    assert step_info["prediction"].tolist() == [3]


def test_env_requires_reset_before_step():
    env = ChemCaptionEnv(molecules=["O"], featurizer=AtomCountFeaturizer())

    with pytest.raises(RuntimeError, match="Call reset"):
        env.step([3])


def test_numeric_reward_metrics():
    featurizer = AtomCountFeaturizer()
    target = FeatureTarget.from_featurizer(featurizer, SMILESMolecule("O"))
    reward_fn = FeatureReward(metric="negative_mae")

    result = reward_fn([2], target)

    assert result.reward == -1.0
    assert result.error == 1.0
    assert result.matches.tolist() == [False]
