Rephrasing Questions
=====================

To increase the diversity of chemical ception we provide an additional
moduel designed to rephrase the questions.

The :ref:`Rephraser` accepts a list of :obj:`chemcaption.featurize.text.Prompt`
objects and rephrases every question. Each question is rephrased 3 times
and samples as many times as you want.

Before rephrasing, we first have to generate some prompts:

.. code-block::python
    from chemcaption.molecules import SMILESMolecule
    from chemcaption.featurize import MolecularFormulaFeaturizer

    smiles = [SMILESMolecule("O"), SMILESMolecule("C1=CC=CC=C1")]
    featurizer = MolecularFormulaFeaturizer()

    prompts = featurizer.text_featurize_many(molecules=smiles)

We then rephrase these prompts using the `SmolLM`_:

.. code-block::python
    from chemcaption.rephraser import SmolLMInstructRePhraser

    model = SmolLMInstructRePhraser(samples=4)
    rephrased = model.rephrase(text)

This will then generate a list of dictionaries, containing the original
questions, answers and alternative rephrased questions.

.. code-block::text
    [{'original': 'Question: What is the molecular formula of the molecule with SMILES O?',
    'answer': 'Answer: H2O',
    'alternative': ['What is the chemical formula of the compound represented by the SMILES string O?',
    'What is the structural formula of the molecule with the SMILES code O?',
    'What is the molecular structure of the compound with the SMILES string O?',
    'What is the molecular structure of the substance with the SMILES code O?',
    'What is the chemical composition of the molecule indicated by the SMILES string O?',
    'What is the chemical composition of the molecule with the SMILES string O?']},
    {'original': 'Question: What is the molecular formula of the molecule with SMILES c1ccccc1?',
    'answer': 'Answer: C6H6',
    'alternative': ['What is the chemical formula of the compound represented by the SMILES string c1ccccc1?',
    'What is the structural formula of the molecule with the SMILES code c1ccccc1?',
    'What is the molecular structure of the compound with the SMILES notation c1ccccc1?',
    'What is the molecular structure of the molecule with the SMILES code c1ccccc1?',
    'What is the chemical composition of the compound with the SMILES notation c1ccccc1?',
    'What is the molecular structure of the substance with the SMILES code c1ccccc1?',
    'What is the chemical composition of the molecule with the SMILES notation c1ccccc1?']}]

To learn more, visit the API documentation :ref:`Rephraser`.

.. _SmolLM: https://huggingface.co/blog/smollm
