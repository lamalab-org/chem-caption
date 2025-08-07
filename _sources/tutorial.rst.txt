Tutorial
========

Now that you have successfully installed the package, we will show you how to use it.

Representing Molecules
-----------------------

Before you can start featurizing your molecules, you need to learn how to create them
first. The molecules can be represented as ``SMILES``, ``SELFIES`` or ``InChI``.

Molecules are represented by their respective classes:

.. code-block:: python

    from chemcaption.molecules import SMILESMolecule, SELFIESMolecule, InChIMolecule

    smiles_molecule = SMILESMolecule('CC(C)NCC(O)COc1cccc2ccccc12.[Cl]')
    selfies_molecule = SELFIESMolecule('[C][C][Branch1][C][C][N][C][C][Branch1][C][O][C][O][C][=C][C][=C][C][=C][C][=C][C][=C][Ring1][#Branch2][Ring1][=Branch1].[ClH0]')
    inchi_molecule = InChIMolecule('InChI=1S/C16H21NO2.ClH/c1-12(2)17-10-14(18)11-19-16-9-5-7-13-6-3-4-8-15(13)16;/h3-9,12,14,17-18H,10-11H2,1-2H3;1H')

We provide an additional dispatcher to handle this easier:

.. code-block:: python

    from chemcaption.molecules import DISPATCH_MAP

    kinds = ['smiles', 'selfies', 'inchi']

    smiles_molecule = DISPATCH_MAP[kinds[0]]('CC(C)NCC(O)COc1cccc2ccccc12.[Cl]')

When working with a set of molecules we provide additional :obj:`chemcaption.molecules.MoleculeCollection` 
object which provides a way to easily handle large sets of molecules.

.. code-block:: python

    from chemcaption.molecules import MoleculeCollection, SMILESMolecule
    
        smiles = ["C1=CC=CC=C1", "O"]
    
        collection = MoleculeCollection(smiles, SMILESMolecule)
    
        for mol in collection:
            ...

For more detailed information, check the API page :ref:`Molecules`.

Featurizing Molecules
----------------------

Here we will show you an example of how to use one of our basic featurizers. Since all
The featurizers inherit from a base class
:obj:`chemcaption.featurize.base.AbstractFeaturizer`, you can use them in a similar way.

We can classify our featurizers into two groups, depending on whether we need to specify some
presets.

Basic Featurizers
~~~~~~~~~~~~~~~~~~

The majority of featurizers can be used as is, without the need to specify anything except
the molecule we want to featurize.

One of the groups of these featurizers is :obj:`chemcaption.featurize.bonds`. Here
We will show you an example of one of them, the
:obj:`chemcaption.featurize.bonds.BondTypeCountFeaturizer`.

.. code-block:: python

    from chemcaption.featurize.bonds import BondTypeCountFeaturizer
    from chemcaption.molecules import SMILESMolecule

    # This featurizer counts the bonds in a molecule
    bt = BondTypeCountFeaturizer()
    molecule = SMILESMolecule("C1=CC=CC=C1")

    # Featurizing and labels
    results = bt.featurize(molecule)
    labels = bt.feature_labels

    # You can also get a dict with corresponding labels
    labeled_results = bt.labeled_featurize(molecule)

The ``featurize()`` returns a numpy array of features, which would be bond counts in our case.

.. code-block:: python

    np.array([[ 0,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  0, 0,  0,  0,  0, 12]])


The ``feature_labels`` is a list of corresponding labels.

.. code-block:: python

    ['num_unspecified_bond', 'num_single_bonds', 'num_double_bonds','num_triple_bonds',
     'num_quadruple_bonds', 'num_quintuple_bonds','num_hextuple_bonds', 'num_oneandahalf_bonds',
     'num_twoandahalf_bonds', 'num_threeandahalf_bonds', 'num_fourandahalf_bonds',
     'num_fiveandahalf_bonds', 'num_aromatic_bonds', 'num_ionic_bonds', 'num_hydrogen_bonds',
     'num_threecenter_bonds', 'num_dativeone_bonds', 'num_dative_bonds', 'num_other_bonds',
     'num_zero_bonds', 'num_bonds']

While the ``labeled_featurize()`` returns a corresponding dictionary.

.. code-block:: python

    {'num_unspecified_bond': 0,
     'num_single_bonds': 6,
     'num_double_bonds': 0,
     'num_triple_bonds': 0,
     'num_quadruple_bonds': 0,
     'num_quintuple_bonds': 0,
     'num_hextuple_bonds': 0,
     'num_oneandahalf_bonds': 0,
     'num_twoandahalf_bonds': 0,
     'num_threeandahalf_bonds': 0,
     'num_fourandahalf_bonds': 0,
     'num_fiveandahalf_bonds': 0,
     'num_aromatic_bonds': 6,
     'num_ionic_bonds': 0,
     'num_hydrogen_bonds': 0,
     'num_threecenter_bonds': 0,
     'num_dativeone_bonds': 0,
     'num_dative_bonds': 0,
     'num_other_bonds': 0,
     'num_zero_bonds': 0,
     'num_bonds': 12}

Since the vast majority of our featurizers can be used without presets, please refer
to the next section to see how to use the featurizers with presets.

Featurizers with Presets
~~~~~~~~~~~~~~~~~~~~~~~~~

Some of the featurizers require us to provide presets; they are made to tell the
featurizer what and how to featurize. For example, presets can define which atoms,
functional groups, or rings you are interested in, and the featurizer can return
their presence, count or percent in a specific molecule.

Now we will show you how to use one of these featurizers to count elements.

.. code-block:: python

    from chemcaption.presets import ORGANIC
    from chemcaption.molecules import SMILESMolecule
    from chemcaption.featurize.composition import ElementCountFeaturizer

    # Molecule we want to featurize
    molecule = SMILESMolecule("C1(Br)=CC=CC=C1Br")

    # We can eather specify the symbol or the full name
    el_count_symbol = ElementCountFeaturizer(['C', 'O', 'H', 'Br'])
    el_count_name = ElementCountFeaturizer(['carbon', 'hydrogen', 'oxygen', 'bromine'])

    # Featurize the molecule
    result1 = el_count_symbol.labeled_featurize(molecule)
    result2 = el_count_name.labeled_featurize(molecule)

Both of the conventions will give us the same result, just with a different
label, corresponding to the name.

.. code-block:: python

    # Results for the featurizer using symbols
    {'num_c_atoms': 6, 'num_o_atoms': 0, 'num_h_atoms': 4, 'num_br_atoms': 2}

    # Results for the featurizer using name
    {'num_carbon_atoms': 6, 'num_hydrogen_atoms': 4, 'num_oxygen_atoms': 0, 'num_bromine_atoms': 2}

Some of the featurizers require more complex presets. For these, we provide a
presets with corresponding smarts, which you can see in more detail on the
:ref:`Presets` page.

Featurizing Many
~~~~~~~~~~~~~~~~~

We can combine any number of featurizers together and run them as a pipeline to
generate as many features as we want to for each molecule.

For this, we utilize :obj:`chemcaption.featurize.base.MultipleFeaturizer` as
follows:

.. code-block:: python

    from chemcaption.presets import ORGANIC
    from chemcaption.molecules import SMILESMolecule
    from chemcaption.featurize.base import MultipleFeaturizer
    from chemcaption.featurize.bonds import BondTypeCountFeaturizer
    from chemcaption.featurize.composition import ElementCountFeaturizer

    molecule = SMILESMolecule("C1(Br)=CC=CC=C1Br")
    featurizer = MultipleFeaturizer([
        ElementCountFeaturizer(['carbon', 'oxygen', 'hydrogen', 'bromine']),
        BondTypeCountFeaturizer()
    ])

    featurizer.labeled_featurize(molecule)

This us with a combined output consisting of features from both featurizers.

.. code-block:: python

    {'num_carbon_atoms': 6,
     'num_hydrogen_atoms': 4,
     'num_oxygen_atoms': 0,
     'num_bromine_atoms': 2,
     'num_unspecified_bond': 0,
     'num_single_bonds': 6,
     'num_double_bonds': 0,
     'num_triple_bonds': 0,
     'num_quadruple_bonds': 0,
     'num_quintuple_bonds': 0,
     'num_hextuple_bonds': 0,
     'num_oneandahalf_bonds': 0,
     'num_twoandahalf_bonds': 0,
     'num_threeandahalf_bonds': 0,
     'num_fourandahalf_bonds': 0,
     'num_fiveandahalf_bonds': 0,
     'num_aromatic_bonds': 6,
     'num_ionic_bonds': 0,
     'num_hydrogen_bonds': 0,
     'num_threecenter_bonds': 0,
     'num_dativeone_bonds': 0,
     'num_dative_bonds': 0,
     'num_other_bonds': 0,
     'num_zero_bonds': 0,
     'num_bonds': 12}

Additionally, we provide a registry as well as registry functions to initialize all the 
featurizers and comparators from a submodule :ref:`Registry`.

Prompts
--------

All of the featurizers can generate prompts simply by calling the
``text_featurize()`` method on the featurizer.

This will create a :obj:`chemcaption.featurize.text.Prompt` object which contains
all the necessary information for the prompts.

.. code-block:: python

    from chemcaption.presets import ORGANIC
    from chemcaption.molecules import SMILESMolecule
    from chemcaption.featurize.composition import ElementCountFeaturizer

    # Molecule we want to featurize
    molecule = SMILESMolecule("C1(Br)=CC=CC=C1Br")

    # We can eather specify the symbol or the full name
    el_count_name = ElementCountFeaturizer(['carbon', 'hydrogen', 'oxygen', 'bromine'])

    # Featurize the molecule
    prompt = el_count_symbol.text_featurize(molecule=molecule)

By calling the ``to_dict()`` method, we can get a dictionary with the prompt
and all the information related to it. For our example, such a prompt would look
like:

.. code-block:: python

    {'representation': 'Brc1ccccc1Br',
    'representation_type': 'SMILES',
    'prompt_template': 'Question: What {VERB} the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} {REPR_STRING}?',
    'completion_template': 'Answer: {COMPLETION}',
    'completion': [6, 4, 0, 2],
    'completion_names': 'atom counts of Carbon, Hydrogen, Oxygen, and Bromine',
    'completion_labels': ['num_carbon_atoms',
    'num_hydrogen_atoms',
    'num_hidrogen_atoms',
    'num_bromine_atoms'],
    'constraint': None,
    'filled_prompt': 'Question: What are the atom counts of Carbon, Hydrogen, Oxygen, and Bromine of the molecule with SMILES Brc1ccccc1Br?',
    'filled_completion': 'Answer: 6, 4, 0, and 2'}

If we are using the :obj:`chemcaption.featurize.base.MultipleFeaturizer`, we
instead get a :obj:`chemcaption.featurize.text.PromptCollection` which holds a PromptCollection
of prompts. Instead of ``to_dict()``, we have to call a ``to_list()`` to generate
a list of prompt dictionaries with all the same information as shown before.

On the next page, we will show you how to create your own featurizers.
