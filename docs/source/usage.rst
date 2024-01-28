Usage
=====
The main idea of chemcaption is to be able to represent molecules in memory, generate molecular fingerprints from these
molecules, and store these fingerprints as either tabular data or graph data (i.e., node/edge attributes). In addition,
we can convert these generated fingerprints as text.

A basic workflow of how to use *chemcaption* can be seen below:

Molecular Representation
------------------------
**Chemcaption** provides multiple means of representing molecules within a program. At present,
molecules can be represented as SMILES molecules, InChI molecules, and SELFIES molecules,
corresponding to the SMILES, InChI and SELFIES string representation systems respectively.

At present, work is ongoing to increase the number of molecular representations supported. As an example,
generating a molecular instance from a SMILES string would proceed like so:

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule
    molecule_string = "CCCC=C" # Pentane
    molecule = SMILESMolecule(representation_string=molecule_string)
..

Similar can be done for other string systems by replacing **SMILESMolecule** and **molecular_string**
with the appropriate class and string respectively.

Molecules as Graphs
-------------------
Molecules can be modeled as graphs too. To do this, there are multiple approaches. The two easiest approaches would be to either:

1. Call the specialized **to_graph** method designed for this (**RECOMMENDED!**) or
2. Pass the molecule into the **MoleculeGraph** class.

Examples:

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule

    molecule_string = "CCCC=C" # Pentane
    molecule = SMILESMolecule(representation_string=molecule_string)
    molecule_graph = molecule.to_graph() # Return molecule as a graph
..

The other approach is displayed below:

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule, MoleculeGraph

    molecule_string = "CCCC=C" # Pentane
    molecule = SMILESMolecule(representation_string=molecule_string)
    molecule_graph = MoleculeGraph(molecule = molecule)
..

The generated graph can then be processed as desired.


Molecule Featurization (Single Molecules)
-----------------------------------------
With the molecular representation ensured, the next step would likely be to featurize a molecule i.e.,
generate molecular fingerprints from the molecule.

To do this, the basic steps would be to:

1. Instatiate the molecule(s) of interest.
2. Instantiate the featurizer of interest (with parameters, if required).
3. Pass the molecule to the **featurize** function.

The **featurize** method returns a numpy array containing the feature of interest.

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule
    from chemcaption.featurize.composition import MolecularMassFeaturizer

    molecule_string = "CCCC=C" # Pentane
    molecule = SMILESMolecule(representation_string=molecule_string) # STEP 1

    featurizer = MolecularMassFeaturizer() # STEP 2

    feature = featurizer.featurize(molecule = molecule) # STEP 3

    print(type(feature))
    print(feature)
..

Molecule Featurization (Batched Molecules)
------------------------------------------
The featurization process is not limited to single molecules alone; featurization can be batched. For this,
we need a sequence or collection of molecular instances and the **featurize_many** featurizer method.

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule
    from chemcaption.featurize.composition import MolecularMassFeaturizer

    molecule_string1 = "CCCC=C"     # Pentane
    molecule_string2 = "[C-]#[O+]"  # Carbon II Oxide
    molecule_string3 = "N#N"        # Nitrogen molecule

    molecule1 = SMILESMolecule(representation_string=molecule_string1) # STEP 1
    molecule2 = SMILESMolecule(representation_string=molecule_string2) # STEP 1
    molecule3 = SMILESMolecule(representation_string=molecule_string3) # STEP 1

    molecules = [
        molecule1,
        molecule2,
        molecule3,
    ]

    featurizer1 = MolecularMassFeaturizer() # STEP 2

    feature = featurizer.featurize_many(molecules = molecules) # STEP 3

    print(type(feature))
    print(feature.shape)
    print(feature)
..


Molecule Featurization (Batched Featurizers)
--------------------------------------------
In addition to batching molecules, featurizers can also be batched. This
allows generation of multiple different fingerprints for multiple different molecules at the same time.
This is done via a special high-level featurize: **MultipleFeaturizer**.

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule

    from chemcaption.featurize.base import MultipleFeaturizer
    from chemcaption.featurize.composition import MolecularMassFeaturizer, AtomCountFeaturizer

    molecule_string1 = "CCCC=C"     # Pentane
    molecule_string2 = "[C-]#[O+]"  # Carbon II Oxide
    molecule_string3 = "N#N"        # Nitrogen molecule

    molecule1 = SMILESMolecule(representation_string=molecule_string1) # STEP 1
    molecule2 = SMILESMolecule(representation_string=molecule_string2) # STEP 1
    molecule3 = SMILESMolecule(representation_string=molecule_string3) # STEP 1

    molecules = [
        molecule1,
        molecule2,
        molecule3,
    ]

    featurizer1 = MolecularMassFeaturizer()                     # STEP 2
    featurizer2 = AtomCountFeaturizer()                         # STEP 2

    featurizers = [featurizer1, featurizer2]                    # STEP 2

    featurizer = MultipleFeaturizer(featurizers = featurizers)  # STEP 2

    feature = featurizer.featurize_many(molecules = molecules) # STEP 3

    print(type(feature))
    print(feature.shape)
    print(feature)
..


Molecule Featurization (Adapted Featurizers)
--------------------------------------------
Some projects require some novel featurization, which is embodied by a function.
This function can be converted into a featurizer of its own by leveraging the **RDKitAdaptor**.

Here, as an example, we define a function which:

1. Takes in a molecular instance,
2. Extracts its molecular string,
3. Tells the number of occurrences of the character **=**, i.e., the number of double bonds in the molecule.

.. code-block:: python

    def carbon_atom_counter_in_string(molecule):
        molecule_string = molecule.representation_string # Get string
        return molecule_string.count("=")
..

This function will then be converted to a featurizer, and the rest of the workflow continues as normal:

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule
    from chemcaption.featurize.adaptor import RDKitAdaptor

    # Convert function to featurizer via RDKitAdaptor
    function_featurizer = RDKitAdaptor(rdkit_function = carbon_atom_counter_in_string)

    # Generate molecule instance
    molecule_string = "N#N"  # Nitrogen molecule
    molecule = SMILESMolecule(representation_string=molecule_string)

    feature = function_featurizer.featurize(molecule = molecule)

    print(type(feature))
    print(feature.shape)
    print(feature)
..


Molecule Featurization (Text)
-----------------------------
The above featurization processes are all valid, with the added functionality of being able to generate the features as
part of text.

Here, for instance, we revisit the example where we convert a function to a featurizer. This time, we generate the
features as text, leveraging the **text_featurize** method.

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule
    from chemcaption.featurize.adaptor import RDKitAdaptor

    # Convert function to featurizer via RDKitAdaptor
    function_featurizer = RDKitAdaptor(rdkit_function = carbon_atom_counter_in_string)

    # Generate molecule instance
    molecule_string = "N#N"  # Nitrogen molecule
    molecule = SMILESMolecule(representation_string=molecule_string)

    feature = function_featurizer.text_featurize(molecule = molecule)

    print(type(feature))
    print(feature.shape)
    print(feature)
..


Molecular Comparison (Single Featurizer)
--------------------------------------------
One of the driving interests of the **chemcaption** project is to improve on efforts to generate graph data
for applications in chemistry. In **chemcaption**, the answer to this is the **Comparator** class.

The **Comparator** class allows the comparison of molecular instances based upon some criteria. Some default
Comparators are implemented in **chemcaption** at the moment. In addition, the Comparator API makes it easy
for any user to define their own Comparator instances.

Utilizing a pre-defined comparator is as simple as:

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule
    from chemcaption.featurize.comparator import AtomCountComparator

    # Convert function to featurizer via RDKitAdaptor
    comparator = AtomCountComparator()

    # Generate molecular instances
    molecule_string1 = "CCCC=C"     # Pentane
    molecule_string2 = "[C-]#[O+]"  # Carbon II Oxide
    molecule_string3 = "N#N"        # Nitrogen molecule

    molecule1 = SMILESMolecule(representation_string=molecule_string1) # STEP 1
    molecule2 = SMILESMolecule(representation_string=molecule_string2) # STEP 1
    molecule3 = SMILESMolecule(representation_string=molecule_string3) # STEP 1

    molecules = [
        molecule1,
        molecule2,
        molecule3,
    ]

    feature = comparator.compare(molecules = molecules)

    print(type(feature))
    print(feature.shape)
    print(feature)
..

As can be attested to, calling the Comparator API is just as straightforward as calling the AbstractFeaturizer API;
just replace the call to the **featurize** method with a call to the **compare** method.


Molecular Comparison (Batched Featurizers & Adapted Comparators)
----------------------------------------------------------------
In addition to comparison on one featurizer, comparison can be carried out between molecules over multiple featurizers.

In this case, all that is needed is to pass a collection of featurizers to the more general **Comparator** constructor.

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule

    from chemcaption.featurize.composition import MolecularMassFeaturizer, AtomCountFeaturizer
    from chemcaption.featurize.base import Comparator

    # Instantiate featurizers for comparison purposes
    featurizer1 = MolecularMassFeaturizer()                     # STEP 2
    featurizer2 = AtomCountFeaturizer()                         # STEP 2

    # Collate featurizers
    featurizers = [featurizer1, featurizer2]

    # Convert function to featurizer via RDKitAdaptor
    comparator = Comparator(featurizers = featurizers)

    # Generate molecular instances
    molecule_string1 = "CCCC=C"     # Pentane
    molecule_string2 = "[C-]#[O+]"  # Carbon II Oxide
    molecule_string3 = "N#N"        # Nitrogen molecule

    molecule1 = SMILESMolecule(representation_string=molecule_string1) # STEP 1
    molecule2 = SMILESMolecule(representation_string=molecule_string2) # STEP 1
    molecule3 = SMILESMolecule(representation_string=molecule_string3) # STEP 1

    molecules = [
        molecule1,
        molecule2,
        molecule3,
    ]

    feature = comparator.compare(molecules = molecules)

    print(type(feature))
    print(feature.shape)
    print(feature)
..


Molecular Comparison (Adapted Comparators)
------------------------------------------------------
To design a specific comparator, all that is needed is to specify what sort of comparison will be carried out.
This specification is packaged as a function: **comparison_func**. This method takes in three main inputs:

* The featurizer of interest,
* The molecules to be compared, and
* a small float for numerical stability (**OPTIONAL!**).

As an example, we will define here a comparator which checks if a set of molecules have the same values for the
same property.

First, we define our comparison function:

.. code-block:: python

    def comparison_function(featurizer, molecules, epsilon = .1):
        results = featurizer.featurize_many(molecules = molecules).flatten().tolist()
        return np.array(len(set(results)) == 1).astype(int)
..

We then use this function to generate our comparator:

.. code-block:: python

    from chemcaption.featurize.base import Comparator

    class MyComparator(Comparator):
        def __init__(self, featurizers, comparison_func):
            super().__init__(comparison_func = comparison_func)
..

With the Comparator class defined, we can move on to actually generating the molecules and comparing them. In this
instance, we wish to compare the molecules to see whether or not they have the same:

1. Molecular mass, and
2. Number of atoms

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule

    from chemcaption.featurize.composition import MolecularMassFeaturizer, AtomCountFeaturizer

    # Instantiate featurizers for comparison purposes
    featurizer1 = MolecularMassFeaturizer()                     # STEP 2
    featurizer2 = AtomCountFeaturizer()                         # STEP 2

    # Collate featurizers
    featurizers = [featurizer1, featurizer2]

    #Instantiate comparator instance
    comparator = MyComparator(featurizers = featurizers, comparison_func = comparison_function)

    # Generate molecular instances
    molecule_string1 = "CCCC=C"     # Pentane
    molecule_string2 = "[C-]#[O+]"  # Carbon II Oxide
    molecule_string3 = "N#N"        # Nitrogen molecule

    molecule1 = SMILESMolecule(representation_string=molecule_string1) # STEP 1
    molecule2 = SMILESMolecule(representation_string=molecule_string2) # STEP 1
    molecule3 = SMILESMolecule(representation_string=molecule_string3) # STEP 1

    molecules = [
        molecule1,
        molecule2,
        molecule3,
    ]

    # Compare molecules
    feature = comparator.compare(molecules = molecules)

    print(feature.shape)
    print(feature)
..


Molecular Comparison (Batched Comparators)
------------------------------------------------------
In addition to the above, and similarly to featurizers, comparators can also be batched.
This allows multiple types of comparisons to be carried out simultaneously.

As an instance, we wish to compare a set of molecules for their number of atoms and their number of valence electrons.
We can do this by leveraging the **MultipleComparator**:

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule

    from chemcaption.featurize.base import MultipleComparator
    from chemcaption.featurize.comparator import AtomCountComparator, ValenceElectronCountComparator

    # Instantiate comparators for comparison purposes
    comparator1 = AtomCountComparator()                # STEP 2
    comparator2 = ValenceElectronCountComparator()     # STEP 2

    # Collate comparators
    comparators = [comparator1, comparator2]

    # Instantiate comparator instance
    mega_comparator = MultipleComparator(featurizers = featurizers)

    # Generate molecular instances
    molecule_string1 = "CCCC=C"     # Pentane
    molecule_string2 = "[C-]#[O+]"  # Carbon II Oxide
    molecule_string3 = "N#N"        # Nitrogen molecule

    molecule1 = SMILESMolecule(representation_string=molecule_string1) # STEP 1
    molecule2 = SMILESMolecule(representation_string=molecule_string2) # STEP 1
    molecule3 = SMILESMolecule(representation_string=molecule_string3) # STEP 1

    molecules = [
        molecule1,
        molecule2,
        molecule3,
    ]

    # Compare molecules
    feature = mega_comparator.compare(molecules = molecules)

    print(feature.shape)
    print(feature)
..
