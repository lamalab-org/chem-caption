# -*- coding: utf-8 -*-

"""Miscellaneous featurizers."""

from typing import List, Optional

import numpy as np
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.molecules import Molecule

# Implemented featurizers

__all__ = [
    "SVGFeaturizer",
]


class SVGFeaturizer(AbstractFeaturizer):
    """Convert molecule instance to SVG image."""

    def __init__(
        self,
        canvas_width: int = 300,
        canvas_height: int = 300,
        include_hydrogens: bool = False,
        highlight_smarts: Optional[str] = None,
    ):
        """Instantiate class.

        Args:
            canvas_width (int): Width of canvas on which to draw molecule. Given in pixels. Defaults to `300`.
            canvas_height (int): Height of canvas on which to draw molecule. Given in pixels. Defaults to `300`.
            include_hydrogens (bool): Include hydrogen atoms in SVG rendering. Defaults to `False`.
            highlight_smarts (Optional[str]): SMARTS pattern for atoms to highlight in SVG rendering. Optional.
        """
        super().__init__()

        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.include_hydrogens = include_hydrogens
        self.highlight_smarts = highlight_smarts

        self._names = [{"noun": "SVG representation"}]

    def _mol_to_svg(self, molecule: Molecule) -> str:
        """Return inline SVG representation for molecule instance.

        Args:
            molecule (Molecule): Molecule instance.

        Return:
            str: SVG string.
        """
        mol = molecule.rdkit_mol if not self.include_hydrogens else molecule.reveal_hydrogens()

        # To highlight atoms or not to highlight...
        highlight_smarts = (
            Chem.MolFromSmarts(self.highlight_smarts)
            if self.highlight_smarts
            else self.highlight_smarts
        )
        highlight_smarts = mol.GetSubstructMatch(highlight_smarts) if highlight_smarts else None

        # Draw molecule
        molecule_drawer = rdMolDraw2D.MolDraw2DSVG(self.canvas_width, self.canvas_height)
        molecule_drawer.DrawMolecule(mol, highlightAtoms=highlight_smarts)

        molecule_drawer.FinishDrawing()

        return molecule_drawer.GetDrawingText()

    def display_molecule(self, molecule: Molecule) -> SVG:
        """Return SVG image for molecule instance.

        Args:
            molecule (Molecule): Molecule instance.

        Return:
            SVG: SVG image.
        """
        svg_string = self._mol_to_svg(molecule=molecule).replace("svg:", "")
        return SVG(svg_string)

    def featurize(self, molecule: Molecule) -> np.ndarray:
        """
        Featurize single molecule instance. Generate SVG representation for molecule.

        Args:
            molecule (Molecule): Molecule representation.

        Returns:
            np.ndarray: Array containing SVG representation in string form.
        """
        return np.array([self._mol_to_svg(molecule=molecule)]).reshape(1, 1)

    def feature_labels(self) -> List[str]:
        """
        Return list of feature labels.

        Args:
            None.

        Returns:
            List[str]: List of feature labels.
        """
        return ["svg_string"]

    def implementors(self) -> List[str]:
        """
        Return list of functionality implementors.

        Args:
            None.

        Returns:
            List[str]: List of implementors.
        """
        return ["Benedict Oshomah Emoekabu", "Kevin Maik Jablonka"]
