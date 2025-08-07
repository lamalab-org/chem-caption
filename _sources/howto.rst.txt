How to Guide
============

What happens if you are not able to find a featurizer that satisfies your needs?
You just make your own one.

Since ChemCaption is a fairly modular framework, you can easily extend it with your
own featurizers and customize them to your own liking.

In this tutorial, we will build a very simple featurizer to just count the length
of the molecule representation string.

To start off, we need to create a skeleton of our feautrizer. There are three main
methods we need to extend; ``implementors()``, ``featurize()`` and ``feature_labels``.

Custom Featurizer Class
------------------------

All of the featurizers are built on the top of the
:obj:`chemcaption.featurize.base.AbstractFeaturizer` class. The skeleton of our
custom featurizer should look something like this:

.. code-block:: python

    from typing import List, Dict

    import numpy as np

    from chemcaption.featurize.base import AbstractFeaturizer
    from chemcaption.molecules import Molecule

    class CustomFeaturizer(AbstractFeaturizer):
        """Our custom featurizer.

        This featurizer just returns the representation string length of
        a molecule.
        """

        def __init__(self):
            super().__init__()

        @property
        def get_names(self) -> List[Dict[str, str]]:
            pass

        @property
        def feature_labels(self) -> List[str]:
            pass

        def featurize(self, molecule: Molecule) -> np.array:
            pass

        def implementors(self) -> List[str]:
            return ['LamaLab']

Now we implement the ``featurize()`` method, which in our case would just
return one integer value, the length of a string.

.. code-block:: python

    def featurize(self, molecule: Molecule) -> np.array:
        return np.array([len(molecule.representation_string)])


The resulting feature needs an accompanying label. To do this, we implement the
``feature_labels`` property method.

.. code-block:: python

    @property
    def feature_labels(self) -> List[str]:
        return ['str_len']


Now that we have a functioning featurizer, we need to provide it with custom
prompt template and names so we can utilize the featurizer to the fullest.

Custom Prompt
--------------

To generate a prompt from our featurizer, we use the ``text_featurize()`` method.
This method creates the :obj:`chemcaption.text.Prompt` object from the class
properties we are going to define shortly.

The first thing we define is the names of the features, since we have only one
the new ``get_names()`` property method will look like the following:

.. code-block:: python

    def get_names(self) -> List[Dict[str, str]]:
        return [{'noun': 'length of the representation string'}]


To finish up the prompt generation, we add the templates and a constraint to our class.

.. code-block:: python

    def __ini__(self):
        def __init__(self):
        # Template for our prompt
        self.prompt_template = (
            "Question: What {VERB} the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} "
            "{REPR_STRING}?"
        )

        # Template for our answer
        self.completion_template = "Answer: {COMPLETION}"

        # Constraint
        self.constraint = "Constraint: Return only a single integer."


The custom variables within these templates are as follows:

* **VERB** - Is either 'is' or 'are' depending on whether we have a single or multiple
    properties.
* **PROPERTY_NAME** - Property names that are split by a space.
* **REPR_SYSTEM** - Is either 'SMILES', 'SELFIES', or 'InChI', depending on the
    molecule representation we used.
* **REPR_STRING** - String representation of the molecule.
* **COMPLETION** - Value of our featurized property.

Finally, after running our custom featurizes for ``C1(Br)=CC=CC=C1Br`` SMILES we
get the following question and answer pair.

.. code-block:: text

    Question: What is the length of the representation string of the molecule with SMILES Brc1ccccc1Br?
    Constraint: Return only a single integer.

    Answer: 12

Full Custom Featurizer
-----------------------

Our new featurizer would then look like this:

.. code-block:: python

    from typing import List, Dict

    import numpy as np

    from chemcaption.featurize.base import AbstractFeaturizer
    from chemcaption.molecules import Molecule

    class CustomFeaturizer(AbstractFeaturizer):
        """Our custom featurizer.

        This featurizer just returns the representation string length of
        a molecule.
        """

        def __init__(self):
            # Template for our prompt
            self.prompt_template = (
                "Question: What {VERB} the {PROPERTY_NAME} of the molecule with {REPR_SYSTEM} "
                "{REPR_STRING}?"
            )

            # Template for our answer
            self.completion_template = "Answer: {COMPLETION}"

            # Constraint
            self.constraint = "Constraint: Return only a single integer."


        @property
        def get_names(self) -> List[Dict[str, str]]:
            return [{'noun': 'length of the representation string'}]

        @property
        def feature_labels(self) -> List[str]:
            return ['str_len']

        def featurize(self, molecule: Molecule) -> np.array:
            return np.array([len(molecule.representation_string)])

        def implementors(self) -> List[str]:
            return ['LamaLab']
