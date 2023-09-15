from chemcaption.featurize.adaptor import ValenceElectronCountAdaptor
from chemcaption.featurize.bonds import (
    BondRotabilityFeaturizer,
    BondTypeCountFeaturizer,
    BondTypeProportionFeaturizer,
)
from chemcaption.featurize.composition import (
    ElementCountFeaturizer,
    ElementMassFeaturizer,
    ElementMassProportionFeaturizer,
    MolecularFormulaFeaturizer,
    MonoisotopicMolecularMassFeaturizer,
)
from chemcaption.featurize.electronicity import (
    HydrogenAcceptorCountFeaturizer,
    HydrogenDonorCountFeaturizer,
)
from chemcaption.featurize.rules import LipinskiViolationCountFeaturizer
from chemcaption.featurize.spatial import (
    InertialShapeFactorFeaturizer,
    EccentricityFeaturizer,
    AsphericityFeaturizer,
    NPRFeaturizer,
    PMIFeaturizer,
)
from chemcaption.featurize.stereochemistry import NumChiralCentersFeaturizer
from chemcaption.featurize.substructure import SMARTSFeaturizer
from chemcaption.presets import ALL_SMARTS
from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.molecules import SMILESMolecule
import fire
import dask.dataframe as dd
from dask import delayed
from typing import Optional, Union
from pathlib import Path
import jsonlines
import time
from functools import partial
from tqdm import tqdm

from dask.distributed import Client


def get_smarts_featurizers():
    featurizers = []
    for name, smarts in ALL_SMARTS.items():
        featurizers.append(SMARTSFeaturizer([smarts], names=[name]))
    return featurizers


FEATURIZER = MultipleFeaturizer(
    get_smarts_featurizers()
    + [
        ValenceElectronCountAdaptor(),
        BondRotabilityFeaturizer(),
        BondTypeCountFeaturizer(),
        BondTypeProportionFeaturizer(),
        MolecularFormulaFeaturizer(),
        MonoisotopicMolecularMassFeaturizer(),
        ElementMassFeaturizer(),
        ElementCountFeaturizer(),
        ElementMassProportionFeaturizer(),
        HydrogenAcceptorCountFeaturizer(),
        HydrogenDonorCountFeaturizer(),
        LipinskiViolationCountFeaturizer(),
        InertialShapeFactorFeaturizer(),
        EccentricityFeaturizer(),
        AsphericityFeaturizer(),
        InertialShapeFactorFeaturizer(),
        NPRFeaturizer(),
        PMIFeaturizer(),
        NumChiralCentersFeaturizer(),
    ]
)


def featurize_smiles(smiles: str):
    try:
        return FEATURIZER.text_featurize(SMILESMolecule(smiles)).to_list()
    except Exception as e:
        print(e)
        return []


# @delayed
def featurize_all_smiles(all_smiles):
    # create a jsonl file with the featurization output

    results = []
    for smiles in all_smiles:
        results.extend(featurize_smiles(smiles))

    with open(f"outname_{all_smiles[0]}.jsonl", "w") as outfile:
        with jsonlines.Writer(outfile) as writer:
            writer.write_all(results)


def chunked_feat_large_df(filepath: Union[str, Path], chunksize: Union[int, str] = ".01MB"):
    # read the csv file, get the SMILES column and in parallel featurize the SMILES

    df = dd.read_csv(filepath, blocksize=chunksize)
    smiles_array = df["SMILES"].to_dask_array(lengths=True)

    smiles_array.map_blocks(featurize_all_smiles).compute()


if __name__ == "__main__":
    client = Client(n_workers=3, processes=True)
    fire.Fire(chunked_feat_large_df)
