# -*- coding: utf-8 -*-

"""Code to extract data from PubChem database."""

import time
from typing import List

import numpy as np
import pandas as pd
import pubchempy as pcp
from selfies import encoder

"""Test data."""


MOLECULAR_BANK = pd.read_json("data/molecular_bank.json", orient="index")

smiles = MOLECULAR_BANK["smiles"].tolist()


class MolecularScraper:
    """Extract data from PubChem database."""

    def __init__(self, smiles_list: List[str]):
        """Initialize class instance."""
        # Out of 2,018 records, 1,450 records were successfully obtained.
        # However, the size of the requests led to the process cutting off.
        self.list = smiles_list
        self.total_len = len(self.list)
        self.properties = [
            "CanonicalSMILES",
            "InChI",
            "InChIKey",
            "MolecularFormula",
            "MolecularWeight",
            "ExactMass",
            "MonoisotopicMass",
            "HBondDonorCount",
            "HBondAcceptorCount",
            "RotatableBondCount",
        ]
        self.column_map = dict(
            zip(
                self.properties,
                [
                    "canon_smiles",
                    "inchi",
                    "inchi_key",
                    "molecular_formular",
                    "molar_mass",
                    "exact_mass",
                    "monoisotopic_mass",
                    "num_hdonors",
                    "num_hacceptors",
                    "num_rotable",
                ],
            )
        )
        self.columns = ["smiles", "selfies"] + [self.column_map[k] for k in self.properties]
        self.filename = "data/legacy/pubchem_response.csv"

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

    def transfer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transfer stacked data to DataFrame of desired attributes.

        Args:
            df (pd.DataFrame): Stacked DataFrame.

        Returns:
            new_df (pd.DataFrame): Formatted DataFrame.
        """
        new_df = pd.DataFrame(columns=self.columns)

        for column in df.columns:
            new_df[column] = df[column].values

        return new_df

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

        try:
            seen = len(pd.read_csv(self.filename))
            self.list = self.list[seen:]
        except FileNotFoundError:
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False)
            seen = 0

        running_seen = seen

        start = time.time()

        for smiles_string in self.list:
            # compound = self.get_compound(smiles_string)
            properties = self.get_properties(smiles_string)
            properties["smiles"] = smiles_string
            properties["selfies"] = encoder(smiles_string)

            dfs.append(properties)
            seen += 1
            running_seen += 1

            if (seen % 50 == 0) or (seen == len(self.list)):
                df_base = pd.read_csv(self.filename)
                df = pd.concat(dfs, axis=0).rename(columns=self.column_map)

                df = self.transfer_data(df)
                df = pd.concat([df_base, df], axis=0)

                df.to_csv(self.filename, na_rep=np.nan, index=False)
                dfs.clear()

                print(f">>> {running_seen}/{self.total_len} compounds scraped!")

                if (time.time() - start) >= 60 & seen >= 390:
                    sleep_time = np.random.randint(low=1, high=5, size=(1,)).item()
                    time.sleep(sleep_time)
                    print(f"\nSleeping for {sleep_time} seconds...\n")
                    start = time.time()
                    seen = 0

        return None


if __name__ == "__main__":
    scraper = MolecularScraper(smiles_list=smiles)
    scraper.run()
