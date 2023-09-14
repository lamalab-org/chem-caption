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
        return None


def featurize_dataframe(
    df: dd.DataFrame, outname: Optional[str] = None, smiles_column: str = "SMILES"
):
    all_smiles = df[smiles_column].dropna().unique()
    print(f"Featurizing {len(all_smiles)} smiles")
    # create a jsonl file with the featurization output
    results = []
    for smiles in all_smiles:
        results.extend(delayed(featurize_smiles)(smiles).compute())

    if outname is None:
        outname = f"featurization_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with open(outname, "w") as outfile:
        with jsonlines.Writer(outfile) as writer:
            writer.write_all(results)

    return results


def chunked_feat_large_df(filepath: Union[str, Path], chunksize: int = 100):
    df = dd.read_csv(filepath, blocksize=chunksize)
    chunks = df.to_delayed()

    for i, chunk in tqdm(enumerate(chunks)):
        curried_featurize = partial(featurize_dataframe, outname=f"featurization_{i}.jsonl")
        delayed(curried_featurize)(chunk).compute()


if __name__ == "__main__":
    fire.Fire(chunked_feat_large_df)
