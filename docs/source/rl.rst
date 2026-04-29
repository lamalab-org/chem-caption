Reinforcement Learning
======================

`chemcaption.rl` provides a Gymnasium-shaped environment over chemcaption
featurizers. The raw featurizer return value is exposed in `info` as
`raw_featurizer_output` and the flattened reward target is exposed as `target`.

Basic Usage
-----------

Create an environment with a molecule list and any chemcaption featurizer. The
environment uses the Gymnasium `reset`/`step` return convention, but
Gymnasium is not required as an installation dependency.

.. code-block:: python

   from chemcaption.featurize.composition import AtomCountFeaturizer
   from chemcaption.rl import ChemCaptionEnv

   env = ChemCaptionEnv(["O"], featurizer=AtomCountFeaturizer())
   observation, info = env.reset()

   assert observation == {
       "representation_system": "SMILES",
       "representation_string": "O",
   }

   target = info["target"]
   labels = info["target_labels"]

   # A policy/model should return predicted feature values in this label order.
   action = target
   observation, reward, terminated, truncated, info = env.step(action)

The default reward is exact feature matching, so the example above receives a
reward of `1.0`. Each episode is single-step: once `step` is called, the
episode terminates.

Accessing Raw Featurizer Output
-------------------------------

For reward functions or target generation that need the unmodified featurizer
return value, read `raw_featurizer_output` from `info`. This is the exact
object returned by `featurizer.featurize(molecule)`.

.. code-block:: python

   observation, info = env.reset()

   raw = info["raw_featurizer_output"]
   flat_target = info["target"]
   labels = info["target_labels"]
   labeled = info["target_labeled"]

   print(raw)
   print(dict(zip(labels, flat_target)))
   print(labeled)

If you want the target in the observation, enable it explicitly:

.. code-block:: python

   env = ChemCaptionEnv(
       ["O"],
       featurizer=AtomCountFeaturizer(),
       include_target_in_observation=True,
   )
   observation, info = env.reset()
   target = observation["target"]

Using Mapping Actions
---------------------

Actions can be either arrays/lists in `target_labels` order or dictionaries
keyed by feature label.

.. code-block:: python

   observation, info = env.reset()

   action = {"num_atoms": 3}
   observation, reward, terminated, truncated, info = env.step(action)

Custom Reward Metrics
---------------------

Use :obj:`chemcaption.rl.FeatureReward` to select a built-in numeric reward
metric.

.. code-block:: python

   from chemcaption.rl import FeatureReward

   env = ChemCaptionEnv(
       ["O"],
       featurizer=AtomCountFeaturizer(),
       reward_fn=FeatureReward(metric="negative_mae"),
   )

   observation, info = env.reset()
   observation, reward, terminated, truncated, info = env.step([2])

Available metrics are `exact`, `negative_mae`, `negative_rmse`, and
`cosine_similarity`.

Combining Featurizers
---------------------

The environment accepts :obj:`chemcaption.featurize.base.MultipleFeaturizer`,
so you can expose multiple package features as one RL target.

.. code-block:: python

   from chemcaption.featurize.base import MultipleFeaturizer
   from chemcaption.featurize.composition import AtomCountFeaturizer, MolecularMassFeaturizer
   from chemcaption.rl import ChemCaptionEnv

   featurizer = MultipleFeaturizer(
       [
           AtomCountFeaturizer(),
           MolecularMassFeaturizer(),
       ]
   )
   env = ChemCaptionEnv(["O", "CCO"], featurizer=featurizer)

   observation, info = env.reset()
   print(info["target_labels"])
   print(info["target"])

Using Registry-Discovered Package Featurizers
---------------------------------------------

To build an environment from registry-discovered featurizers, use
`make_package_featurizer` or `make_env`.

.. code-block:: python

   from chemcaption.rl import ChemCaptionEnv, make_package_featurizer

   featurizer = make_package_featurizer(groups=["composition", "rules"])
   env = ChemCaptionEnv(["O", "CCO"], featurizer=featurizer)

Registry discovery may initialize expensive featurizers depending on the
selected groups. For lightweight examples and tests, pass explicit featurizer
instances instead.

.. automodule:: chemcaption.rl
   :members:
