"""Reinforcement-learning utilities for chemcaption featurizers.

The environment in this module follows the Gymnasium `reset`/`step` return
shape without requiring Gymnasium as a dependency. Each episode presents one
molecule, evaluates one action against the raw featurizer target, and then
terminates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from chemcaption.featurize.base import AbstractFeaturizer, MultipleFeaturizer
from chemcaption.molecules import AbstractMolecule, DISPATCH_MAP, Molecule

__all__ = [
    "ChemCaptionEnv",
    "FeatureReward",
    "FeatureTarget",
    "RewardResult",
    "make_env",
    "make_package_featurizer",
]

def _flat_array(values: object) -> np.ndarray:
    """Return a flattened array without changing the raw object."""
    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape((1,))
    return array.reshape((-1,))


def _as_float_array(values: np.ndarray) -> np.ndarray | None:
    """Return float view when every value is numeric, otherwise `None`."""
    try:
        return np.asarray(values, dtype=float).reshape((-1,))
    except (TypeError, ValueError):
        return None


def _to_python(value: object) -> object:
    """Convert numpy scalar containers to plain Python objects for info dicts."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _coerce_molecule(molecule: str | Molecule, representation: str) -> Molecule:
    """Convert strings to chemcaption molecule objects."""
    if isinstance(molecule, AbstractMolecule):
        return molecule

    if isinstance(molecule, str):
        try:
            molecule_class = DISPATCH_MAP[representation.lower()]
        except KeyError as exc:
            valid = ", ".join(sorted(DISPATCH_MAP))
            raise ValueError(
                f"Unsupported representation '{representation}'. Use one of: {valid}."
            ) from exc
        return molecule_class(molecule)

    raise TypeError("Molecules must be chemcaption molecule instances or representation strings.")


def _action_values(action: Action, labels: tuple[str, ...]) -> np.ndarray:
    """Normalize an action into a flat array aligned with `labels`."""
    if isinstance(action, FeatureTarget):
        values = action.values
    elif isinstance(action, Mapping):
        if len(set(labels)) != len(labels):
            raise ValueError("Mapping actions require unique feature labels.")

        missing = [label for label in labels if label not in action]
        if missing:
            raise ValueError(f"Action mapping is missing feature label(s): {missing}.")

        values = [action[label] for label in labels]
    else:
        values = action

    flat = _flat_array(values)
    if len(flat) != len(labels):
        raise ValueError(
            f"Action has {len(flat)} feature value(s), but the target has {len(labels)}."
        )
    return flat


@dataclass(frozen=True)
class FeatureTarget:
    """Featurizer target for one molecule.

    Attributes:
        molecule: Molecule used to generate the target.
        raw: Exact object returned by `featurizer.featurize(molecule)`.
        values: Flattened view of `raw` aligned with `labels`.
        labels: Feature labels reported by the featurizer.
    """

    molecule: Molecule
    raw: object
    values: np.ndarray
    labels: tuple[str, ...]

    @classmethod
    def from_featurizer(cls, featurizer: AbstractFeaturizer, molecule: Molecule) -> "FeatureTarget":
        """Build a target from a chemcaption featurizer and molecule.

        Args:
            featurizer: Featurizer used to generate the raw target.
            molecule: Molecule passed to the featurizer.

        Returns:
            Feature target containing the raw and flattened values.
        """
        raw = featurizer.featurize(molecule)
        values = _flat_array(raw)
        labels = tuple(featurizer.feature_labels)

        if len(values) != len(labels):
            raise ValueError(
                f"{featurizer.__class__.__name__} returned {len(values)} value(s), "
                f"but exposes {len(labels)} feature label(s)."
            )

        return cls(molecule=molecule, raw=raw, values=values, labels=labels)

    @property
    def representation_system(self) -> str:
        """Return the molecule representation system."""
        return self.molecule.get_representation()

    @property
    def representation_string(self) -> str:
        """Return the molecule representation string."""
        return self.molecule.representation_string

    @property
    def labeled(self) -> dict[str, object]:
        """Return feature values keyed by label.

        For duplicate labels, later values overwrite earlier values. Use
        `items` when duplicate labels must be preserved.
        """
        return {
            label: _to_python(value)
            for label, value in zip(self.labels, self.values)
        }

    @property
    def items(self) -> list[dict[str, object]]:
        """Return label/value pairs while preserving order and duplicate labels."""
        return [
            {"label": label, "value": _to_python(value)}
            for label, value in zip(self.labels, self.values)
        ]

    def observation(self, include_target: bool = False) -> dict[str, object]:
        """Return an observation dictionary for policies.

        Args:
            include_target: Whether to include target fields in the observation.

        Returns:
            Observation dictionary for the current molecule.
        """
        observation = {
            "representation_system": self.representation_system,
            "representation_string": self.representation_string,
        }

        if include_target:
            observation.update(
                {
                    "target": self.values.copy(),
                    "target_labels": list(self.labels),
                    "raw_featurizer_output": self.raw,
                }
            )

        return observation

    def info(self) -> dict[str, object]:
        """Return metadata and raw target values for reward calculation.

        Returns:
            Info dictionary for the current molecule target.
        """
        return {
            "molecule": self.molecule,
            "representation_system": self.representation_system,
            "representation_string": self.representation_string,
            "target": self.values.copy(),
            "target_labels": list(self.labels),
            "target_labeled": self.labeled,
            "target_items": self.items,
            "raw_featurizer_output": self.raw,
        }


Action = np.ndarray | Sequence[object] | Mapping[str, object] | FeatureTarget


@dataclass(frozen=True)
class RewardResult:
    """Reward calculation result.

    Attributes:
        reward: Scalar reward value.
        metric: Reward metric name.
        prediction: Flattened action values aligned with `labels`.
        target: Flattened target values aligned with `labels`.
        labels: Feature labels used for alignment.
        matches: Per-feature exact-match indicators.
        error: Numeric error value when the metric reports one.
    """

    reward: float
    metric: str
    prediction: np.ndarray
    target: np.ndarray
    labels: tuple[str, ...]
    matches: np.ndarray
    error: float | None = None

    def info(self) -> dict[str, object]:
        """Return reward diagnostics for an environment `info` dict.

        Returns:
            Dictionary containing reward diagnostics.
        """
        info = {
            "reward_metric": self.metric,
            "prediction": self.prediction.copy(),
            "feature_matches": self.matches.copy(),
            "num_features": len(self.labels),
            "num_matches": int(np.sum(self.matches)),
        }

        if self.error is not None:
            info["reward_error"] = self.error

        return info


class FeatureReward:
    """Compare an action against a raw featurizer target.

    Supported metrics are `exact`, `negative_mae`, `negative_rmse`, and
    `cosine_similarity`. Exact numeric comparisons use `numpy.isclose` with the
    configured tolerances.
    """

    _METRICS = {"exact", "negative_mae", "negative_rmse", "cosine_similarity"}

    def __init__(
        self,
        metric: str = "exact",
        absolute_tolerance: float = 1e-8,
        relative_tolerance: float = 1e-5,
    ):
        """Initialize the reward function.

        Args:
            metric: Reward metric to apply.
            absolute_tolerance: Absolute tolerance for numeric exact matches.
            relative_tolerance: Relative tolerance for numeric exact matches.
        """
        if metric not in self._METRICS:
            raise ValueError(
                f"Unsupported reward metric '{metric}'. Use one of {sorted(self._METRICS)}."
            )

        self.metric = metric
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance

    def __call__(self, action: Action, target: FeatureTarget) -> RewardResult:
        """Return reward diagnostics for an action and target.

        Args:
            action: Predicted feature values.
            target: Featurizer target for the current molecule.

        Returns:
            Reward result with diagnostics.
        """
        prediction = _action_values(action, target.labels)
        target_values = target.values
        target_numeric = _as_float_array(target_values)
        prediction_numeric = _as_float_array(prediction)

        if target_numeric is not None and prediction_numeric is not None:
            matches = np.isclose(
                prediction_numeric,
                target_numeric,
                atol=self.absolute_tolerance,
                rtol=self.relative_tolerance,
            )
            reward, error = self._numeric_reward(prediction_numeric, target_numeric, matches)
        else:
            if self.metric != "exact":
                raise ValueError(
                    f"Reward metric '{self.metric}' requires numeric prediction and target values."
                )

            matches = np.array(
                [actual == expected for actual, expected in zip(prediction, target_values)]
            )
            reward = float(np.mean(matches))
            error = None

        return RewardResult(
            reward=reward,
            metric=self.metric,
            prediction=prediction,
            target=target_values.copy(),
            labels=target.labels,
            matches=matches,
            error=error,
        )

    def _numeric_reward(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        matches: np.ndarray,
    ) -> tuple[float, float | None]:
        """Compute numeric reward from normalized arrays."""
        if self.metric == "exact":
            return float(np.mean(matches)), None

        diff = prediction - target
        if self.metric == "negative_mae":
            error = float(np.mean(np.abs(diff)))
            return -error, error

        if self.metric == "negative_rmse":
            error = float(np.sqrt(np.mean(np.square(diff))))
            return -error, error

        prediction_norm = float(np.linalg.norm(prediction))
        target_norm = float(np.linalg.norm(target))
        if prediction_norm == 0.0 and target_norm == 0.0:
            return 1.0, None
        if prediction_norm == 0.0 or target_norm == 0.0:
            return 0.0, None

        reward = float(np.dot(prediction, target) / (prediction_norm * target_norm))
        return reward, None


class ChemCaptionEnv:
    """Single-step RL environment over chemcaption molecule featurizers.

    The environment emits a molecule observation on `reset`. The next `step`
    expects an action containing predicted feature values. The environment then
    compares that action to the raw featurizer output, returns the reward, and
    terminates the episode.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        molecules: Iterable[str | Molecule],
        featurizer: AbstractFeaturizer,
        representation: str = "smiles",
        reward_fn: FeatureReward | None = None,
        sampler: str = "sequential",
        include_target_in_observation: bool = False,
        seed: int | None = None,
    ):
        """Initialize the environment.

        Args:
            molecules: Molecules or representation strings to sample.
            featurizer: Featurizer used to create reward targets.
            representation: Representation system for string molecules.
            reward_fn: Reward function. Defaults to exact feature matching.
            sampler: Molecule sampling strategy, either `sequential` or `random`.
            include_target_in_observation: Whether observations include target data.
            seed: Random seed for the sampler, if provided.
        """
        if sampler not in {"sequential", "random"}:
            raise ValueError("sampler must be either 'sequential' or 'random'.")

        self.molecules = [_coerce_molecule(molecule, representation) for molecule in molecules]
        if not self.molecules:
            raise ValueError("ChemCaptionEnv requires at least one molecule.")

        if not isinstance(featurizer, AbstractFeaturizer):
            raise TypeError(
                "featurizer must be an instance of "
                "chemcaption.featurize.base.AbstractFeaturizer."
            )

        self.featurizer = featurizer
        self.reward_fn = reward_fn or FeatureReward()
        self.representation = representation
        self.sampler = sampler
        self.include_target_in_observation = include_target_in_observation
        self.current_index: int | None = None
        self.current_target: FeatureTarget | None = None
        self._terminated = True
        self._next_index = 0
        self._rng = np.random.default_rng(seed)

        # Kept for Gymnasium-style wrappers without requiring gymnasium.spaces.
        self.action_space = None
        self.observation_space = None

    @property
    def raw_featurizer_output(self) -> object | None:
        """Return the current raw featurizer output, if an episode is active."""
        if self.current_target is None:
            return None
        return self.current_target.raw

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Seed the random sampler.

        Args:
            seed: Random seed.

        Returns:
            Gym-compatible seed list.
        """
        self._rng = np.random.default_rng(seed)
        return [seed]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Start a new single-molecule episode.

        Args:
            seed: Random seed for the sampler, if provided.
            options: Reset parameters, if provided. Use `index` to select a molecule.

        Returns:
            Observation and info dictionaries.
        """
        if seed is not None:
            self.seed(seed)

        index = self._select_index(options=options)
        self.current_index = index
        self.current_target = FeatureTarget.from_featurizer(self.featurizer, self.molecules[index])
        self._terminated = False

        return self._observation(), self._info()

    def step(
        self,
        action: Action,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        """Evaluate one feature prediction action and terminate the episode.

        Args:
            action: Predicted feature values.

        Returns:
            Observation, reward, terminated flag, truncated flag, and info dictionary.
        """
        if self.current_target is None or self._terminated:
            raise RuntimeError("Call reset() before step().")

        reward_result = self.reward_fn(action, self.current_target)
        self._terminated = True

        info = self._info()
        info.update(reward_result.info())
        return self._observation(), reward_result.reward, True, False, info

    def render(self, mode: str = "ansi") -> str | None:
        """Render the current molecule representation.

        Args:
            mode: Render mode, either `ansi` or `human`.

        Returns:
            Rendered string for `ansi` mode, otherwise `None`.
        """
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode '{mode}'.")

        if self.current_target is None:
            rendered = ""
        else:
            rendered = (
                f"{self.current_target.representation_system}: "
                f"{self.current_target.representation_string}"
            )

        if mode == "human":
            print(rendered)
            return None
        return rendered

    def close(self) -> None:
        """Release environment resources."""
        return None

    def _select_index(self, options: Mapping[str, object] | None = None) -> int:
        """Select the molecule index for the next episode."""
        if options and "index" in options:
            index = int(options["index"])
            if index < 0 or index >= len(self.molecules):
                raise IndexError(f"Molecule index {index} is out of range.")
            return index

        if self.sampler == "random":
            return int(self._rng.integers(len(self.molecules)))

        index = self._next_index
        self._next_index = (self._next_index + 1) % len(self.molecules)
        return index

    def _observation(self) -> dict[str, object]:
        """Return the current observation."""
        if self.current_target is None:
            raise RuntimeError("Call reset() before requesting an observation.")
        return self.current_target.observation(
            include_target=self.include_target_in_observation
        )

    def _info(self) -> dict[str, object]:
        """Return the current info dictionary."""
        if self.current_target is None:
            raise RuntimeError("Call reset() before requesting info.")

        info = self.current_target.info()
        info["index"] = self.current_index
        return info


def make_package_featurizer(
    groups: Sequence[str] | None = None,
    extra_featurizers: Sequence[AbstractFeaturizer] | None = None,
) -> MultipleFeaturizer:
    """Create a `MultipleFeaturizer` from registry-discovered featurizers.

    Registry discovery can initialize computationally expensive featurizers, so
    this helper imports the registry lazily and is not used by `ChemCaptionEnv`
    unless explicitly requested.

    Args:
        groups: Registry group names to include, if provided.
        extra_featurizers: Additional featurizers to append, if provided.

    Returns:
        Combined featurizer containing the selected featurizers.
    """
    from chemcaption.featurize import registry

    registry_groups = {
        "bonds": registry.BONDS_FEATURIZERS,
        "composition": registry.COMPOSITION_FEATURIZERS,
        "electronicity": registry.ELECTRONICITY_FEATURIZERS,
        "miscellaneous": registry.MISCELLANEOUS_FEATURIZERS,
        "reaction": registry.REACTION_FEATURIZERS,
        "rules": registry.RULES_FEATURIZERS,
        "spatial": registry.SPATIAL_FEATURIZERS,
        "stereochemistry": registry.STEREOCHEMISTRY_FEATURIZERS,
        "substructure": registry.SUBSTRUCTURE_FEATURIZERS,
        "symmetry": registry.SYMMETRY_FEATURIZERS,
    }

    selected_groups = (
        list(registry_groups) if groups is None else [group.lower() for group in groups]
    )
    unknown = sorted(set(selected_groups) - set(registry_groups))
    if unknown:
        valid = ", ".join(sorted(registry_groups))
        raise ValueError(f"Unknown featurizer group(s): {unknown}. Use one of: {valid}.")

    featurizers: list[AbstractFeaturizer] = []
    for group in selected_groups:
        featurizers.extend(registry_groups[group])

    if extra_featurizers is not None:
        featurizers.extend(extra_featurizers)

    return MultipleFeaturizer(featurizers=featurizers)


def make_env(
    molecules: Iterable[str | Molecule],
    featurizer: AbstractFeaturizer | None = None,
    representation: str = "smiles",
    **kwargs: object,
) -> ChemCaptionEnv:
    """Create a `ChemCaptionEnv`.

    If `featurizer` is omitted, the environment uses all registry-discovered
    package featurizers via `make_package_featurizer`.

    Args:
        molecules: Molecules or representation strings to sample.
        featurizer: Featurizer for reward targets, if provided.
        representation: Representation system for string molecules.
        **kwargs: Additional arguments forwarded to `ChemCaptionEnv`.

    Returns:
        Configured chemcaption environment.
    """
    if featurizer is None:
        featurizer = make_package_featurizer()

    return ChemCaptionEnv(
        molecules=molecules,
        featurizer=featurizer,
        representation=representation,
        **kwargs,
    )
