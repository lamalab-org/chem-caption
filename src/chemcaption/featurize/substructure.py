# -*- coding: utf-8 -*-

"""Featurizers describing the presence of substructures in a molecule."""



class SMARTSFeaturizer(AbstractFeaturizer):
    """A featurizer for molecular substructure search via SMARTS."""

    def __init__(
        self,
        count: bool = True,
        names: Optional[Union[str, List[str]]] = "rings",
        smarts: Optional[List[str]] = None,
    ):
        """
        Initialize class.

        Args:
            count (bool): If set to True, count pattern frequency. Otherwise, only encode presence. Defaults to True.
            names (Optional[Union[str, List[str]]]): Preset name(s) of the substructures encoded by the SMARTS strings.
                Predefined presets can be specified as strings, and can be one of:
                    - `heterocyclic`,
                    - `rings`,
                    - `amino`,
                    - `scaffolds`,
                    - `warheads` or
                    - `organic`.
                Defaults to `rings`.
            smarts (Optional[List[str]]): SMARTS strings that are matched with the molecules. Defaults to None.
        """
        super().__init__()

        if isinstance(names, str):
            try:
                names, smarts = SMARTSPreset(names).preset
            except KeyError:
                raise KeyError(
                    f"`{names}` preset not defined. \
                    Use `heterocyclic`, `rings`, 'amino`, `scaffolds`, `warheads`, or `organic`"
                )
        else:
            assert bool(names) == bool(
                smarts
            ), "Both `names` and `smarts` must either be or not be provided."
            assert len(names) == len(
                smarts
            ), "Both `names` and `smarts` must be lists of the same length."

        self.names = names
        self.smarts = smarts
        self.count = count

        self.prefix = ""
        self.suffix = "_count" if count else "_presence"
        self.label = [self.prefix + element.lower() + self.suffix for element in self.names]

    @property
    def preset(self) -> Dict[str, List[str]]:
        """Get molecular preset. Getter method.

        Args:
            None.

        Returns:
            (Dict[str, List[str]]): Dictionary of substance names and substance SMARTS strings.
        """
        return dict(names=self.names, smarts=self.smarts)

    @preset.setter
    def preset(
        self,
        new_preset: Optional[Union[str, Dict[str, List[str]]]],
    ) -> None:
        """Set molecular preset. Setter method.

        Args:
            new_preset (Optional[Union[str, Dict[str, List[str]]]]): New preset of interest.
                Could be a:
                    (str): string representing new predefined preset.
                    (Dict[str, List[str]]): dictionary.
                        Keys: `name` and `smarts`.
                        Values: list of substance names and list of corresponding SMARTS strings.

        Returns:
            None
        """
        if new_preset is not None:
            if isinstance(new_preset, str):
                names, smarts = SMARTSPreset(preset=new_preset).preset()
            elif isinstance(new_preset, (tuple, list)):
                names = new_preset[0]
                smarts = new_preset[1]
            else:
                names = new_preset["names"]
                smarts = new_preset["smarts"]

            self.names = names
            self.smarts = smarts

            self.label = [self.prefix + element.lower() + self.suffix for element in self.names]
        else:
            self.names = None
            self.smarts = None
            self.label = [None]
        return

    def featurize(
        self, molecule: Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]
    ) -> np.array:
        """
        Return integers representing the frequency or presence of molecular patterns in a molecule.

        Args:
            molecule (Union[SMILESMolecule, InChIMolecule, SELFIESMolecule]): Molecule representation.

        Returns:
            (np.array): Array containing integer counts/signifier of pattern presence.
        """
        if self.count:
            results = [
                len(molecule.rdkit_mol.GetSubstructMatches(rdkit.Chem.MolFromSmarts(smart)))
                for smart in self.smarts
            ]
        else:
            results = [
                int(molecule.rdkit_mol.HasSubstructMatch(rdkit.Chem.MolFromSmarts(smart)))
                for smart in self.smarts
            ]

        return np.array(results).reshape((1, -1))
