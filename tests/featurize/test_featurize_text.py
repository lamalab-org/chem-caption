# -*- coding: utf-8 -*-

"""Unit tests for `chemcaption.featurize.text` submodule."""

from chemcaption.featurize.text import Prompt, PromptCollection

__all__ = [
    "test_prompt_container",
]


def test_prompt_container():
    prompt_1 = Prompt(
        completion="Answer: 30",
        representation="c1ccccc1",
        representation_type="SMILES",
        completion_type="int",
        completion_names="valence_electron_count",
        completion_labels="Valence Electron Count",
        prompt_template="Question: What is the number of valence electrons of the molecule with SMILES {REPR_STRING}?",
        completion_template="Answer: {COMPLETION}",
        constraint=None,
    )

    prompt_2 = Prompt(
        completion="Answer: 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,",
        representation="c1ccccc1",
        representation_type="SMILES",
        completion_type="list",
        completion_names="bond_type_proportion",
        completion_labels="Bond Type Proportion",
        prompt_template="Question: What is the proportion of unspecified, single, double, triple, quadruple, quintuple, hextuple, one-and-a-half, two-and-a-half, three-and-a-half, four-and-a-half, five-and-a-half, aromatic, ionic, hydrogen, three-center, dative one-electron, dative two-electron, other, and zero-order bonds in the molecule with SMILES {REPR_STRING}?",
        completion_template="Answer: {REPR_STRING}",
        constraint="Constraint: Return a list of comma separated floats.",
    )

    prompt_container = PromptCollection([prompt_1, prompt_2])

    elements = prompt_container.to_list()
    assert len(elements) == 2
    assert isinstance(elements[0], dict)
