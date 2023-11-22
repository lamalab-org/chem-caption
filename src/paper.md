---
title: 'Chemcaption: A Python package for molecular captioning and cheminformatics data generation'
tags:
  - Python
  - chemistry
  - digital chemistry
  - molecular captioning
  - molecular fingerprint
  - graph
  - Large language models (LLM)
  - Graph neural networks (GNN)
  - pretraining
authors:
  - name: Benedict Oshomah Emoekabu
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
  - name: Kevin M Jablonka
    equal-contrib: true
    affiliation: "Jablonka Group, Freidrich Schiller University"
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

# Statement of need

`Chemcaption` is a cheminformatics package written in `Python` and designed for the tasks of:
+ `molecular captioning` (for `LLM` pretraining) and
+ `molecular fingerprint generation` (for `GNN` pretraining)

This makes `Chemcaption` suitable for generating and composing cheminformatics data in form of either text corpora,
tabular data, molecular visuals (e.g., SVG images) or as graph modalities.

`Chemcaption` provides a convenient wrapper (and connector) around (and between) other cheminformatics packages
like `RDKit`, `pymatgen`, and `morfeus`. A good number of these tools, like `RDKit`, are written in lower-level
languages like `C/C++`, which gives them the advantage of speed. The ability to access these tools via a unified
interface provided by `Chemcation` does two things:

+ `Unified chemistry`: It provides an easier means for leveraging a wide array of available tools and libraries for 
cheminformatics tasks in machine learning.
+ `Digital chemistry`: It improves on the ongoing efforts to digitize chemistry, much as many other areas of human 
endeavor have long been digitized.

`Chemcaption` is very simple and intuitive to use, with the main requirement being the provision of molecular strings.
These strings can be provided as `SMILES`, `SMARTS`, `SELFIES`, or `InChI`, with upcoming implementation allowing for
other molecular representation systems like `TUCAN`. Beyond this, meaningful and sensible presets are made available 
for users, with further customization available for more experienced users.

The `Chemcaption` API is expressly designed to integrate as seamlessly as possible with the APIs of other available
tools such as:
+ `Mofdscribe`,
+ `DScribe`,
+ `Matminer`,
+ `AutoMatminer` etcetera.

Given that there are already a good number of alternatives, the pertinent question to ask is: Why `Chemcaption`?

`Chemcaption` is designed to cater to some shortcomings observed within the aforementioned alternatives. These are:

+ `Data generation`: Most of the alternatives focus on data retrieval, not actual data generation.
+ `Data independence`: Focusing on data generation decouples the ML process from 3rd-party data / sources of truth.
+ `Big data generation & organization`: `Chemcaption` is focused on generating and organizing data at large scales.

There are three main operations that `Chemcaption` performs, with all other operations stemming from these:

+ Generate molecular fingerprints,
+ Combine and compose the generated fingerprints in different modalities, and
+ Compare molecules based on generated and/or composed data.

All of these, especially the comparison process in particular can be carried out based on a large number of criteria.
`Chemcaption` is compositional, and easily customizable, making it easy for users to either select pre-determined
criteria and/or presets, or implement theirs. In fact, it is this versatility that gives `Chemcaption` it's specific
promise for graph data generation.

The design of `Chemcaption` was chosen to ensure ease-of-use for a wide demographic of users: hobbyists, students, and
professional researchers. At present, `Chemcaption` has already proven to be of worth for data generation; an excess of
`40 GB` (and counting) of cheminformatics data has been generated for open-source use, with more on the way. Efforts
are also ongoing to improve and expand the library with more and better generative utilities.



`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References