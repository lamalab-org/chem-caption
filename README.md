<h1 align="center" display="inline-block">
  <img align="center" src="docs/source/_static/logo.png" width=75> 
  <span> ChemCaption </span>
</h1>

<p align="center">
    <a href="https://github.com/lamalab-org/chem-caption/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/lamalab-org/chem-caption/actions/workflows/tests.yml/badge.svg" />
    </a>
    <!-- <a href="https://pypi.org/project/chemcaption">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/chemcaption" />
    </a> -->
    <!-- <a href="https://pypi.org/project/chemcaption">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/chemcaption" />
    </a> -->    
    <a href="https://github.com/lamalab-org/chem-caption/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/chemcaption" />
    </a>
    <!-- <a href='https://chemcaption.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/chemcaption/badge/?version=latest' alt='Documentation Status' />
    </a> -->
    <!-- <a href="https://codecov.io/gh/kjappelbaum/chem-caption/branch/main">
        <img src="https://codecov.io/gh/kjappelbaum/chem-caption/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>   -->
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com/kjappelbaum/chem-caption/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

Caption molecules and materials for pretraining for neural networks.

## 💪 Getting Started

ChemCaption is a tool designed to generate prompts for molecular features to train neural networks. 

Here is a quick example of one of the featurizers designed to count the number of elements in a molecule.

```python
from chemcaption.presets import ORGANIC
from chemcaption.molecules import SMILESMolecule
from chemcaption.featurize.composition import ElementCountFeaturizer

# Molecule we want to featurize
molecule = SMILESMolecule("C1(Br)=CC=CC=C1Br")

# We can eather specify the symbol or the full name
el_count_name = ElementCountFeaturizer(['carbon', 'hydrogen', 'oxygen', 'bromine'])

# Featurize the molecule
prompt = el_count_name.text_featurize(molecule=molecule)
```

The generate prompt has the following QA pair.

```text
Question: What are the atom counts of Carbon, Hydrogen, Hidrogen, and Bromine of the molecule with SMILES Brc1ccccc1Br?
Answer: 6, 4, 0, and 2
```

For more details and all other available featurizers please visit the [documentation]().

## 🚀 Installation

The most recent release can be installed from PyPI with:

```bash
pip install chemcaption
```

The most recent code and data can be installed directly from GitHub with:

```bash
pip install git+https://github.com/lamalab-org/chem-caption
```

Some of the ChemCaption featurizers are dependent on [morfeus](https://digital-chemistry-laboratory.github.io/morfeus/index.html) and might require additional dependencies to be installed. You can see all the optional dependencies for morfeus-ml [here](https://digital-chemistry-laboratory.github.io/morfeus/installation.html)

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/kjappelbaum/chem-caption/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/lamalab-org/chem-caption
$ cd chem-caption
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `nox` with `pip install nox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ nox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/kjappelbaum/chem-caption/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/lamalab-org/chem-caption
$ cd chem-caption
$ nox --session docs
$ open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`nox` with `pip install nox`, the commands for making a new release are contained within the `finish` environment
in `noxfile.py`. Run the following from the shell:

```shell
$ nox --session finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/chemcaption/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `nox -e bumpversion -- minor` after.
</details>

