# -*- coding: utf-8 -*-

"""Code to extract data from PubChem database."""

from typing import List

import numpy as np
import pandas as pd
import pubchempy as pcp

"""Test data."""


MOLECULAR_BANK = pd.read_json("data/molecular_bank.json", orient="index")

smiles = MOLECULAR_BANK["smiles"].tolist()


class MolecularScraper:
    """Extract data from PubChem database."""

    def __init__(self, smiles_list: List[str]):
        """Initialize class instance."""
        ## Out of 2,018 records, 1,450 records were successfully obtained.
        ## However, the size of the requests led to the process cutting off.
        self.list = smiles_list[1450:]
        self.properties = [
            "MolecularFormula",
            "MolecularWeight",
            "ExactMass",
            "MonoisotopicMass",
            "HBondDonorCount",
            "HBondAcceptorCount",
            "RotatableBondCount",
        ]
        self.columns = self.properties + ["smiles"]
        self.filename = "data/pubchem_response.csv"

    def get_properties(self, substance: str) -> pd.DataFrame:
        """
        Get properties of a substance.

        Args:
            substance (str): SMILES string.

        Returns:
            properties (pd.DataFrame): Dataframe containing properties of `substance`.
        """
        properties = pcp.get_properties(
            namespace="smiles", as_dataframe=True, identifier=substance, properties=self.properties
        )
        return properties

    def get_compound(self, smile_string: str) -> pcp.Compound:
        """
        Get the compound object representing a SMILES molecular string.

        Args:
            smile_string (str): SMILES representation string.

        Returns:
            compound (pcp.Compound): Compound instance.
        """
        compound = pcp.get_compounds(smile_string, "smiles")
        return compound

    def run(self) -> None:
        """Perform final data extraction. Persist data to storage."""
        dfs = list()
        seen = 0
        # df = pd.DataFrame(columns=self.columns)
        # df.to_csv(self.filename, index=False)

        for smiles_string in self.list:
            # compound = self.get_compound(smiles_string)
            properties = self.get_properties(smiles_string)
            properties["smiles"] = smiles_string
            dfs.append(properties)
            seen += 1
            if (len(dfs) % 50 == 0) or (seen == len(self.list)):
                dfs_1 = pd.read_csv(self.filename)
                dfs_ = pd.concat(dfs, axis=0)
                dfs_ = pd.concat([dfs_1, dfs_], axis=0)
                dfs_.to_csv(self.filename, na_rep=np.nan, index=False)
                dfs.clear()

                print(f"{seen}/{len(self.list)} compounds scraped!")

        return None


if __name__ == "__main__":
    scraper = MolecularScraper(smiles_list=smiles)
    scraper.run()
