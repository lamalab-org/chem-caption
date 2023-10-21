from pathlib import Path
from typing import Union

import dask.dataframe as dd
import fire
import jsonlines
import numpy as np
from dask.distributed import Client
from rdkit import Chem
from selfies import encoder

from chemcaption.featurize.adaptor import ValenceElectronCountAdaptor
from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.featurize.bonds import (
    BondOrderFeaturizer,
    BondRotabilityFeaturizer,
    BondTypeCountFeaturizer,
    BondTypeProportionFeaturizer,
    DipoleMomentsFeaturizer,
    RotableBondCountFeaturizer,
)
from chemcaption.featurize.composition import (
    AtomCountFeaturizer,
    DegreeOfUnsaturationFeaturizer,
    ElementCountFeaturizer,
    ElementCountProportionFeaturizer,
    ElementMassFeaturizer,
    ElementMassProportionFeaturizer,
    MolecularFormulaFeaturizer,
    MonoisotopicMolecularMassFeaturizer,
)
from chemcaption.featurize.electronicity import (
    AtomChargeFeaturizer,
    AtomElectrophilicityFeaturizer,
    AtomNucleophilicityFeaturizer,
    HOMOEnergyFeaturizer,
    HydrogenAcceptorCountFeaturizer,
    HydrogenDonorCountFeaturizer,
    LUMOEnergyFeaturizer,
    MoleculeElectrofugalityFeaturizer,
    MoleculeElectrophilicityFeaturizer,
    MoleculeNucleofugalityFeaturizer,
    MoleculeNucleophilicityFeaturizer,
)
from chemcaption.featurize.rules import (
    GhoseFilterFeaturizer,
    LeadLikenessFilterFeaturizer,
    LipinskiViolationCountFeaturizer,
)
from chemcaption.featurize.spatial import (
    AsphericityFeaturizer,
    EccentricityFeaturizer,
    InertialShapeFactorFeaturizer,
    NPRFeaturizer,
    PMIFeaturizer,
)
from chemcaption.featurize.stereochemistry import NumChiralCentersFeaturizer
from chemcaption.featurize.substructure import SMARTSFeaturizer
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule
from chemcaption.presets import ALL_SMARTS


def get_smarts_featurizers():
    featurizers = []
    for name, smarts in ALL_SMARTS.items():
        featurizers.append(SMARTSFeaturizer([smarts], names=[name]))
    return featurizers


FEATURIZER = MultipleFeaturizer(
    get_smarts_featurizers()
    + [
        AtomCountFeaturizer(),
        ValenceElectronCountAdaptor(),
        BondRotabilityFeaturizer(),
        BondTypeCountFeaturizer(),
        BondTypeProportionFeaturizer(),
        BondOrderFeaturizer(),
        RotableBondCountFeaturizer(),
        DipoleMomentsFeaturizer(),
        MolecularFormulaFeaturizer(),
        MonoisotopicMolecularMassFeaturizer(),
        ElementMassFeaturizer(),
        ElementCountFeaturizer(),
        ElementMassProportionFeaturizer(),
        ElementCountProportionFeaturizer(),
        DegreeOfUnsaturationFeaturizer(),
        HydrogenAcceptorCountFeaturizer(),
        HydrogenDonorCountFeaturizer(),
        HOMOEnergyFeaturizer(),
        LUMOEnergyFeaturizer(),
        AtomChargeFeaturizer(),
        AtomNucleophilicityFeaturizer(),
        AtomElectrophilicityFeaturizer(),
        MoleculeNucleophilicityFeaturizer(),
        MoleculeElectrophilicityFeaturizer(),
        MoleculeNucleofugalityFeaturizer(),
        MoleculeElectrofugalityFeaturizer(),
        LipinskiViolationCountFeaturizer(),
        GhoseFilterFeaturizer(),
        LeadLikenessFilterFeaturizer(),
        EccentricityFeaturizer(),
        AsphericityFeaturizer(),
        InertialShapeFactorFeaturizer(),
        NPRFeaturizer(),
        PMIFeaturizer(),
        NumChiralCentersFeaturizer(),
    ]
)


def to_smiles_molecule(smiles: str):
    return SMILESMolecule(smiles)


def to_selfies_molecule(smiles: str):
    return SELFIESMolecule(encoder(smiles))


def to_inchi_molecule(smiles: str):
    return InChIMolecule(Chem.MolToInchi(Chem.MolFromSmiles(smiles)))


def convert_molecules(smiles: str):
    conversion_functions = [to_smiles_molecule, to_selfies_molecule, to_inchi_molecule]

    random_conv = np.random.choice(conversion_functions)

    return random_conv(smiles)


def featurize_smiles(smiles: str):
    try:
        return FEATURIZER.text_featurize(convert_molecules(smiles)).to_list()
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


def chunked_feat_large_df(filepath: Union[str, Path], chunksize: Union[int, str] = "12GB"):
    # read the csv file, get the SMILES column and in parallel featurize the SMILES

    df = dd.read_csv(filepath, blocksize=chunksize)
    smiles_array = df["SMILES"].to_dask_array(lengths=True)

    smiles_array.map_blocks(featurize_all_smiles).compute()


if __name__ == "__main__":
    # cluster = SLURMCluster(
    #     cores=24,
    #     processes=6,
    #     memory="16GB",
    #     account="co_laika",
    #     queue="savio2_bigmem",
    #     job_script_prologue=[
    #         'export LANG="en_US.utf8"',
    #         'export LANGUAGE="en_US.utf8"',
    #         'export LC_ALL="en_US.utf8"',
    #     ],
    #     job_extra_directives=['--qos="savio_lowprio"'],
    # )
    client = Client(n_workers=4, processes=True)
    fire.Fire(chunked_feat_large_df)
