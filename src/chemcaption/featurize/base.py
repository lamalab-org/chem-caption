# -*- coding: utf-8 -*-

"""Abstract base class and wrappers for featurizers."""

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rdkit
from colorama import Fore
from frozendict import frozendict
from morfeus import SASA, XTB
from morfeus.conformer import Conformer, ConformerEnsemble
from rdkit import Chem
from scipy.spatial import distance_matrix

from chemcaption.featurize.text import Prompt, PromptCollection
from chemcaption.featurize.utils import cached_conformer, join_list_elements
from chemcaption.molecules import Molecule

# Implemented abstract and high-level classes

__all__ = [
    "AbstractFeaturizer",  # Featurizer base class.
    "MorfeusFeaturizer",
    "AbstractComparator",
    "MultipleFeaturizer",  # Combines multiple featurizers.
    "Comparator",  # Class for comparing featurizer results amongst molecules.
    "MultipleComparator",  # Higher-level Comparator. Returns lower-level Comparator instances.
    "PERIODIC_TABLE",  # Periodic table
]

PERIODIC_TABLE = rdkit.Chem.GetPeriodicTable()  # Periodic table


"""Abstract class"""


class AbstractFeaturizer(ABC):
    """Abstract base class for lower level Featurizers."""

    def __init__(self):
        """Initialize class. Initialize periodic table."""
        self.prompt_template = "Question: What is the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} {REPR_STRING}?"
        self.completion_template = "Answer: {COMPLETION}"
        self._names = []
        self.constraint = None

    def get_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        return self._names

    @abstractmethod
    def featurize(self, molecule: Molecule) -> np.array:
        """Featurize single Molecule instance."""
        raise NotImplementedError

    def featurize_many(self, molecules: List[Molecule]) -> np.array:
        """
        Featurize a sequence of Molecule objects.

        Args:
            molecules (Sequence[Molecule]):
                A sequence of molecule representations.

        Returns:
            (np.array): An array of features for each molecule instance.
        """
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.featurize, molecules))

        return np.concatenate(results)

    def text_featurize(
        self,
        molecule: Molecule,
    ) -> Prompt:
        """Embed features in Prompt instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (Prompt): Instance of Prompt containing relevant information extracted from `molecule`.
        """
        completion = self.featurize(molecule=molecule).tolist()

        completion_type = [type(c) for c in completion]
        representation = molecule.representation_string
        representation_type = molecule.__repr__().split("Mole")[0]

        completion_labels = self.feature_labels()

        completion_name = self.get_names()[0]["noun"]

        return Prompt(
            completion=join_list_elements(completion),
            completion_type=completion_type,
            representation=representation,
            representation_type=representation_type,
            completion_names=completion_name,
            completion_labels=completion_labels,
            prompt_template=self.prompt_template,
            completion_template=self.completion_template,
            constraint=self.constraint,
        )

    def text_featurize_many(
        self,
        molecules: List[Molecule],
    ) -> List[Prompt]:
        """Embed features in Prompt instance for multiple molecules.

        Args:
            molecules (List[Molecule]): A list of molecule representations.

        Returns:
            (List[Prompt]): List of Prompt instances containing relevant information extracted from each
                molecule in `molecules`.
        """
        return [self.text_featurize(molecule=molecule) for molecule in molecules]

    @abstractmethod
    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        raise NotImplementedError

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
        raise NotImplementedError

    def feature_names(self) -> List[Dict[str, str]]:
        """Return feature names.

        Args:
            None.

        Returns:
            (List[Dict[str, str]]): List of names for extracted features according to parts-of-speech.
        """
        return self._names

    def citations(self):
        """Return citation for this project."""
        raise NotImplementedError


class MorfeusFeaturizer(AbstractFeaturizer):
    """Abstract featurizer for morfeus-generated features."""

    def __init__(
        self,
        conformer_generation_kwargs: Optional[Dict[str, Any]] = None,
        morfeus_kwargs: Optional[Dict[str, Any]] = None,
        qc_optimize: bool = False,
        aggregation: Optional[Union[str, List[str]]] = None,
    ):
        """Instantiate class.

        Args:
            conformer_generation_kwargs (Optional[Dict[str, Any]]): Configuration for conformer generation.
            morfeus_kwargs (Optional[Dict[str, Any]]): Keyword arguments for morfeus computation.
            qc_optimize (bool): Run QCEngine optimization harness. Defaults to `False`.
            aggregation (Optional[Union[str, List[str]]]): Aggregation to use on generated descriptors. Defaults to `None`.
        """
        super().__init__()
        self._conf_gen_kwargs = (
            frozendict(conformer_generation_kwargs)
            if conformer_generation_kwargs
            else frozendict({})
        )
        self.morfeus_kwargs = frozendict(morfeus_kwargs) if morfeus_kwargs else frozendict({})
        self.qc_optimize = qc_optimize

        # Function map for supported aggregations
        self.aggregation_func = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "min": np.min,
            "max": np.max,
        }

        self._acceptable_aggregations = list(self.aggregation_func.keys()) + [None]

        if type(aggregation) is str:
            aggregation = aggregation.lower()
        elif type(aggregation) is list:
            aggregation = [agg.lower() for agg in aggregation]
        else:
            pass

        self.aggregation = aggregation

        assert self._check_aggregation(
            self.aggregation
        ), "Invalid aggregation. Available aggregations are {}".format(
            self._acceptable_aggregations
        )

    def _check_aggregation(self, aggregations: Union[str, List[str]]) -> bool:
        """Ensure supported aggregations are provided.

        Args:
            None.

        Returns:
            bool: Authenticity of provided aggregations.
        """
        if isinstance(aggregations, str) or aggregations is None:
            aggregations = [
                aggregations,
            ]

        return all([(agg in self._acceptable_aggregations) for agg in aggregations])

    def _get_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        """Return conformer for molecule.

        Args:
            mol (Chem.Mol): rdkit Molecule.

        Returns:
            (Chem.Mol): Molecule instance embedded with conformers.
        """
        smiles = Chem.MolToSmiles(mol)
        return cached_conformer(smiles, self._conf_gen_kwargs)

    @staticmethod
    def _parse_indices(
        atom_indices: Union[int, List[int]], as_range: bool = False
    ) -> Tuple[Sequence, bool]:
        """Preprocess atom indices.

        Args:
            atom_indices (Union[int, List[int]]): Range of atoms to calculate areas for. Either:
                - an integer,
                - a list of integers, or
                - a two-tuple of integers representing lower index and upper index.
            as_range (bool): Use `atom_indices` parameter as a range of indices or not. Defaults to `False`
        """
        if as_range:
            if isinstance(atom_indices, int):
                atom_indices = range(1, atom_indices + 1)

            elif len(atom_indices) == 2:
                if atom_indices[0] > atom_indices[1]:
                    raise IndexError(
                        "`atom_indices` parameter should contain two integers as (lower, upper) i.e., [10, 20]"
                    )
                atom_indices = range(atom_indices[0], atom_indices[1] + 1)

            else:
                as_range = False
                print(
                    Fore.RED
                    + "UserWarning: List of integers passed to `atom_indices` parameter. `as_range` parameter will be refactored to False."
                    + Fore.RESET
                )

        else:
            if isinstance(atom_indices, int):
                atom_indices = [atom_indices]

        return atom_indices, as_range

    def fit_on_bond_counts(
        self, molecules: Union[List[Molecule], Tuple[Molecule], Molecule]
    ) -> int:
        """Fit instance on molecule collection.

        Args:
            molecules (Union[List[Molecule], Tuple[Molecule], Molecule]): List of molecular instances.

        Returns:
            int: Maximum number of bonds in any molecule passed in to featurizer.
        """
        molecules = [molecules] if not isinstance(molecules, list) else molecules
        bond_counts = [self._count_bonds(molecule=molecule) for molecule in molecules]
        return max(bond_counts)

    @staticmethod
    def fit_on_atom_counts(molecules: Union[List[Molecule], Tuple[Molecule], Molecule]) -> int:
        """Fit instance on molecule collection.

        Args:
            molecules (Union[List[Molecule], Tuple[Molecule], Molecule]): List of molecular instances.

        Returns:
            int: Maximum number of atoms in any molecule passed in to featurizer.
        """
        molecules = [molecules] if not isinstance(molecules, list) else molecules
        atom_counts = [molecule.reveal_hydrogens().GetNumAtoms() for molecule in molecules]
        return max(atom_counts)

    @staticmethod
    def _count_bonds(molecule: Molecule) -> int:
        """Helper function to count the number of bonds in a molecule.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            int: Integer representing the number of bonds in a molecule.
        """
        bonds = list(molecule.reveal_hydrogens().GetBonds())
        return len(bonds)

    def _get_element_coordinates(self, molecule: Molecule) -> Tuple[np.array, np.array]:
        """Return appropriate morfeus instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (Tuple[np.array, np.array]): Tuple containing (a). atoms and (b). corresponding coordinates in molecule.
        """
        molecule = self._get_conformer(molecule.reveal_hydrogens())

        elements = np.array(
            [PERIODIC_TABLE.GetElementSymbol(atom.GetAtomicNum()) for atom in molecule.GetAtoms()]
        )
        coordinates = molecule.GetConformer().GetPositions()

        return elements, coordinates

    def _get_morfeus_instance(
        self, molecule: Molecule, morpheus_instance: str = "xtb"
    ) -> Union[SASA, XTB]:
        """Return appropriate morfeus instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.
            morpheus_instance (str): Type of morfeus instance. Can take on either `xtb` or `sasa`. Defaults to `xtb`.

        Returns:
            (Union[SASA, XTB]): Appropriate morfeus instance.
        """
        if morpheus_instance.lower() not in ["xtb", "sasa"]:
            raise Exception(
                "`morpheus_instance` parameter must take on either `xtb` or `sasa` as value."
            )

        return (
            self._get_sasa_instance(molecule)
            if morpheus_instance.lower() == "sasa"
            else self._get_xtb_instance(molecule)
        )

    def _get_xtb_instance(self, molecule: Molecule) -> XTB:
        """Return appropriate morfeus instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (XTB): Appropriate morfeus XTB instance.
        """
        elements, coordinates = self._get_element_coordinates(molecule)

        return XTB(elements, coordinates, "1")

    def _get_sasa_instance(self, molecule: Molecule) -> SASA:
        """Return appropriate morfeus instance for feature generation.

        Args:
            molecule (Molecule): Molecular instance.

        Returns:
            (SASA): Appropriate morfeus SASA instance.
        """
        elements, coordinates = self._get_element_coordinates(molecule)

        return SASA(elements, coordinates, **self.morfeus_kwargs)

    @staticmethod
    def _optimize_molecule_geometry(
        molecule: Molecule,
        optimization_method: str = "GFN2-xTB",
        procedure: str = "geometric",
        rmsd_method: str = "spyrmsd",
    ) -> ConformerEnsemble:
        """Generate conformers and optimize them in 3D space.

        Args:
            molecule (Molecule): Molecular instance.
            optimization_method (str): Method to be applied for geometric optimization. Defaults to `GFN2-xTB`.
            procedure (str): QC engine optimization procedure. Defaults to `geometric`.
            rmsd_method (str): Base method for conformer pruning w.r.t RMSD property.

        Returns:
            (ConformerEnsemble): An ensemble of generated conformers.
        """
        string = Chem.MolToSmiles(molecule.rdkit_mol)
        # Generate and optimize an ensemble of conformers
        conformer_ensemble = ConformerEnsemble.from_rdkit(string)
        conformer_ensemble.optimize_qc_engine(
            program="xtb", model={"method": optimization_method}, procedure=procedure
        )
        conformer_ensemble = conformer_ensemble.prune_rmsd(method=rmsd_method)
        conformer_ensemble.sort()
        return conformer_ensemble.update_mol()

    def _generate_conformers(
        self,
        molecule: Molecule,
        num_conformers: int = 1,
        optimization_method: str = "GFN2-xTB",
        procedure: str = "geometric",
        rmsd_method: str = "spyrmsd",
    ) -> List[Conformer]:
        """Generate conformers and optimize them in 3D space.

        Args:
            molecule (Molecule): Molecular instance.
            num_conformers (int): Number of conformers to return after optimization.
            optimization_method (str): Method to be applied for geometric optimization. Defaults to `GFN2-xTB`.
            procedure (str): QC engine optimization procedure. Defaults to `geometric`.
            rmsd_method (str): Base method for conformer pruning w.r.t RMSD property.

        Returns:
            (List[Chem.Mol]): A list of generated conformers.
        """
        conformer_ensemble = self._optimize_molecule_geometry(
            molecule=molecule,
            optimization_method=optimization_method,
            procedure=procedure,
            rmsd_method=rmsd_method,
        )
        conformer_ensemble.sort()  # Sort conformers based on energy levels
        print("There are", len(conformer_ensemble.conformers), "conformers")
        return conformer_ensemble.conformers[:num_conformers]

    def _generate_conformer(
        self,
        molecule: Molecule,
        optimization_method: str = "GFN2-xTB",
        procedure: str = "geometric",
        rmsd_method: str = "spyrmsd",
    ) -> Molecule:
        """Generate a single conformer.

        Args:
            molecule (Molecule): Molecular instance.
            optimization_method (str): Method to be applied for geometric optimization. Defaults to `GFN2-xTB`.
            procedure (str): QC engine optimization procedure. Defaults to `geometric`.
            rmsd_method (str): Base method for conformer pruning w.r.t RMSD property.

        Returns:
            (Molecule): Molecular instance.
        """
        conformer_ensemble = self._optimize_molecule_geometry(
            molecule=molecule,
            optimization_method=optimization_method,
            procedure=procedure,
            rmsd_method=rmsd_method,
        )

        print(f"{len(conformer_ensemble.conformers)} conformer(s) generated!")

        try:
            molecule.rdkit_mol = conformer_ensemble.mol
        except:
            print(
                Fore.RED
                + "Wholescale conformer embedding failed. Embedding conformers individually...\n"
                + Fore.RESET
            )
            molecule.rdkit_mol = molecule.reveal_hydrogens()

            num_embedded = 0
            conformers = list(conformer_ensemble.mol.GetConformers())

            for conf in conformers:
                try:
                    molecule.rdkit_mol.AddConformer(conf)
                    num_embedded += 1
                except:
                    pass

            message = f"{num_embedded}/{len(conformers)} conformers embedded successfully!\n"
            print(message + "=" * 70 + "\n")

        return molecule

    def _match_coordinates(self, coordinates: np.array):
        """Match atom coordinates.

        Args:
            atom (Chem.Atom): Atom instance.
            coordinates (np.array): Array of atom coordinates.

        Returns:

        """
        Chem.MolToBlock()
        Chem.Compute2DCoords()

    def _track_atom_identity(
        self, molecule: Molecule, max_index: int = 1
    ) -> List[Union[int, float]]:
        """Ensure atom identities are tracked irrespective of atom arrangement in molecule.

        Args:
            molecule (Molecule): Molecular instance.
            max_index (int): Maximum number of atoms/bonds to consider for identity tracking.

        Returns:
            List[Union[int, float]]: Atomic numbers of atoms in `molecule` arranged by index.
        """
        elements, coordinates = self._get_element_coordinates(molecule=molecule)
        atoms = molecule.get_atoms(hydrogen=True)
        atomic_numbers = [atom.GetAtomicNum() for atom in atoms]
        if (max_index - len(atomic_numbers)) > 0:
            atomic_numbers += [0 for _ in range(max_index - len(atomic_numbers))]
        elif (max_index - len(atomic_numbers)) < 0:
            atomic_numbers = atomic_numbers[:max_index]
        return atomic_numbers

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class AbstractComparator(ABC):
    """Abstract base class for Comparator objects."""

    def __init__(self):
        """Initialize class. Initialize periodic table."""
        self.template = None
        self._names = []

    @abstractmethod
    def featurize(self, molecules: List[Molecule]) -> np.array:
        """Featurize multiple Molecule instances."""
        raise NotImplementedError

    @abstractmethod
    def compare(self, molecules: List[Molecule]) -> np.array:
        """Compare features from multiple molecular instances. 1 if all molecules are similar, else 0."""
        raise NotImplementedError

    @abstractmethod
    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        raise NotImplementedError

    @abstractmethod
    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for extracted features.
        """
        raise NotImplementedError

    def citations(self):
        """Return citation for this project."""
        return None


"""Higher-level featurizers."""


class MultipleFeaturizer(AbstractFeaturizer):
    """A featurizer to combine featurizers."""

    def __init__(self, featurizers: Optional[List[AbstractFeaturizer]] = None):
        """Initialize class instance.

        Args:
            featurizers (Optional[List[AbstractFeaturizer]]): A list of featurizer objects. Defaults to `None`.

        """
        super().__init__()

        self.featurizers = featurizers

    def featurize(self, molecule: Molecule) -> np.array:
        """
        Featurize a molecule instance. Returns results from multiple lower-level featurizers.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            features (np.array), array shape [1, N]:
                Array containing features extracted from molecule.
                `N` >= the number of featurizers passed to MultipleFeaturizer.
        """
        features = [
            feature for f in self.featurizers for feature in f.featurize(molecule).flatten()
        ]

        return np.array(features).reshape((1, -1))

    def text_featurize(
        self,
        molecule: Molecule,
    ) -> PromptCollection:
        """Embed features in Prompt instance.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            (PromptCollection): Instance of Prompt containing relevant information extracted from `molecule`.
        """
        return PromptCollection([f.text_featurize(molecule=molecule) for f in self.featurizers])

    def featurize_many(self, molecules: List[Molecule]) -> np.array:
        """
        Featurize a sequence of Molecule objects.

        Args:
            molecules (List[Molecule]): A sequence of molecule representations.

        Returns:
            (np.array): An array of features for each molecule instance.
        """
        results = [f.featurize_many(molecules=molecules) for f in self.featurizers]
        return np.concatenate(results, axis=1)

    def feature_labels(self) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            List[str]: List of labels for all features extracted by all featurizers.
        """
        labels = [label for f in self.featurizers for label in f.feature_labels()]

        return labels

    def fit_on_featurizers(self, featurizers: Optional[List[AbstractFeaturizer]] = None):
        """Fit MultipleFeaturizer instance on lower-level featurizers.

        Args:
            featurizers (Optional[List[AbstractFeaturizer]]): List of lower-level featurizers. Defaults to `None`.

        Returns:
            self : Instance of self with state updated.
        """
        # Type check for AbstractFeaturizer instances
        for ix, featurizer in enumerate(featurizers):
            # Each featurizer must be specifically of type AbstractFeaturizer

            if not isinstance(featurizer, AbstractFeaturizer):
                raise ValueError(
                    f"`{featurizer.__class__.__name__}` instance at index {ix} is not of type `AbstractFeaturizer`."
                )

        self.featurizers = featurizers

        print(f"`{self.__class__.__name__}` instance fitted with {len(featurizers)} featurizers!\n")
        self.label = self.feature_labels()

        self.prompt_template = [featurizer.prompt_template for featurizer in featurizers]
        self.completion_template = [featurizer.completion_template for featurizer in featurizers]

        self._names = [featurizer._names for featurizer in featurizers]

        return self

    def generate_data(self, molecules: List[Molecule], metadata: bool = False) -> pd.DataFrame:
        """Convert generated feature array to DataFrame.

        Args:
            molecules (List[Molecule]): Collection of molecular instances.
            metadata (bool): Include extra molecule information.
                Defaults to `False`.

        Returns:
            (pd.DataFrame): DataFrame generated from feature array.
        """
        features = self.featurize_many(molecules=molecules)

        if metadata:
            extra_columns = ["representation_system", "representation_string"]
            extra_features = np.array(
                [[mol.get_representation(), mol.representation_string] for mol in molecules]
            ).reshape((-1, 2))

            features = np.concatenate([extra_features, features], axis=-1)
        else:
            extra_columns = []

        columns = extra_columns + self.feature_labels()

        return pd.DataFrame(data=features, columns=columns)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class Comparator(AbstractComparator):
    """Compare molecules based on featurizer outputs."""

    def __init__(self, featurizers: Optional[List[AbstractFeaturizer]] = None):
        """Instantiate class.

        Args:
            featurizers (Optional[List[AbstractFeaturizer]]): List of featurizers to compare over. Defaults to `None`.

        """
        super().__init__()
        self.featurizers = None
        self.fit_on_featurizers(featurizers=featurizers)

    def fit_on_featurizers(self, featurizers: Optional[List[AbstractFeaturizer]] = None):
        """Fit Comparator instance on lower-level featurizers.

        Args:
            featurizers (Optional[List[AbstractFeaturizer]]): List of lower-level featurizers. Defaults to `None`.

        Returns:
            self : Instance of self with state updated.
        """
        if featurizers is None:
            self.featurizers = featurizers
            return self
        # Type check for AbstractFeaturizer instances
        for ix, featurizer in enumerate(featurizers):
            # Each featurizer must be specifically of type AbstractFeaturizer

            if not isinstance(featurizer, AbstractFeaturizer):
                raise ValueError(
                    f"`{featurizer.__class__.__name__}` instance at index {ix} is not of type `AbstractFeaturizer`."
                )

        self.featurizers = featurizers

        return self

    def __str__(self):
        """Return string representation.

        Args:
            None.

        Returns:
            (str): String representation of `self`.
        """
        return self.__class__.__name__

    def _compare_on_featurizer(
        self,
        featurizer: AbstractFeaturizer,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.array:
        """Return results of molecule feature comparison between molecule instance pairs.

        Args:
            featurizer (AbstractFeaturizer): Featurizer to compare on.
            molecules (List[Molecule]):
                List containing a pair of molecule instances.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            (np.array): Comparison results. 1 if all extracted features are equal, else 0.
        """
        batch_results = featurizer.featurize_many(molecules=molecules)

        distance_results = distance_matrix(batch_results, batch_results)

        return (np.mean(distance_results) <= epsilon).astype(int).reshape((1, -1))

    def featurize(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.array:
        """
        Featurize multiple molecule instances.

        Extract and return comparison between molecular instances. 1 if similar, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            (np.array): Array containing extracted features with shape `(1, N)`,
                where `N` is the number of featurizers provided at initialization time.
        """
        results = [
            self._compare_on_featurizer(featurizer=featurizer, molecules=molecules, epsilon=epsilon)
            for featurizer in self.featurizers
        ]
        return np.concatenate(results, axis=-1)

    def feature_labels(
        self,
    ) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for all features extracted by all featurizers.
        """
        labels = []
        for featurizer in self.featurizers:
            labels += featurizer.feature_labels()

        labels = [label + "_similarity" for label in labels]

        return labels

    def compare(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.array:
        """
        Compare features from multiple molecular instances. 1 if all molecules are similar, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            (np.array): Array containing comparison results with shape `(1, N)`,
                where `N` is the number of featurizers provided at initialization time.
        """
        return self.featurize(molecules=molecules, epsilon=epsilon)

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]


class MultipleComparator(Comparator):
    """A Comparator to combine Comparators."""

    def __init__(self, comparators: Optional[List[Comparator]] = None):
        """Instantiate class.

        Args:
            comparators (Optional[List[Comparator]]): List of Comparator instances. Defaults to `None`.
        """
        super().__init__()

        self.comparators = None

        self.fit_on_comparators(comparators=comparators)  # If all comparators pass the check

    def fit_on_comparators(self, comparators: Optional[List[Comparator]] = None):
        """Fit MultipleComparator instance on lower-level comparators.

        Args:
            comparators (Optional[List[Comparator]]): List of lower-level comparators. Defaults to `None`.

        Returns:
            (self) : Instance of self with state updated.
        """
        if comparators is None:
            self.comparators = comparators
            return self

        # Type check for Comparator instances
        for ix, comparator in enumerate(comparators):
            # Each comparator must be specifically of types:
            #   `Comparator` or `MultipleComparator` (allows for nesting purposes)

            if not isinstance(comparator, Comparator):
                raise ValueError(
                    f"`{comparator.__class__.__name__}` instance at index {ix}",
                    " is not of type `Comparator` or `MultipleComparator`.",
                )
        self.comparators = comparators

        return self

    def featurize(
        self,
        molecules: List[Molecule],
        epsilon: float = 0.0,
    ) -> np.array:
        """
        Compare features from multiple molecular Comparators. 1 if all molecules are similar, else 0.

        Args:
            molecules (List[Molecule]): Molecule instances to be compared.
            epsilon (float): Small float. Precision bound for numerical inconsistencies. Defaults to 0.0.

        Returns:
            (np.array): Array containing comparison results with shape `(1, N)`,
                where `N` is the number of Comparators provided at initialization time.
        """
        features = [
            comparator.featurize(molecules=molecules, epsilon=epsilon)
            for comparator in self.comparators
        ]

        return np.concatenate(features, axis=-1)

    def feature_labels(
        self,
    ) -> List[str]:
        """Return feature label(s).

        Args:
            None.

        Returns:
            (List[str]): List of labels for all features extracted by all comparators.
        """
        labels = []
        for comparator in self.comparators:
            labels += comparator.feature_labels()

        return labels

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            (List[str]): List of implementors.
        """
        return ["Benedict Oshomah Emoekabu"]
