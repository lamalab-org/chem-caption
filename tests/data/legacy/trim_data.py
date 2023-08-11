# -*- coding: utf-8 -*-

"""Script for trimming test data."""

import pandas as pd
import os

BASE_DIR = os.getcwd()

OLD_PATH = os.path.join(BASE_DIR.replace("legacy", ""), "merged_pubchem_response.csv")
NEW_PATH = os.path.join(BASE_DIR.replace("legacy", ""), "pubchem_response.csv")


def data_trimmer(length=100):

    data = pd.read_csv(OLD_PATH).iloc[:length, :]

    data.to_csv(NEW_PATH, index=False)

    return


if __name__ == "__main__":
    data_trimmer(length=49)


