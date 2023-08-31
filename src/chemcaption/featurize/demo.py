# -*- coding: utf-8 -*-

"""Demo script."""

from chemcaption.molecules import SMILESMolecule
from chemcaption.featurize.composition import ElementMassFeaturizer
from chemcaption.featurize.comparator import IsoelectronicComparator

featurizer = IsoelectronicComparator() # Minimal inputs

molecule = SMILESMolecule("CC(C)NCC(O)COc1cccc2ccccc12.[Cl]")

molecule_strings = ["[N-3]", "[O-2]"]
molecules = [SMILESMolecule(m) for m in molecule_strings]

extracted_feature = featurizer.compare(molecules=molecules)
labels = featurizer.feature_labels()

print("Extracted features: \n\n", extracted_feature)
print("\nExtracted labels: \n\n", labels)