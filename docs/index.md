# Graphas

A Python library for **speech graph analysis**: it represents a speech
transcription as a directed word graph and extracts the structural features used
to quantify how organized or fragmented a discourse is. The approach is
language-agnostic and needs no linguistic pre-processing or reference corpus.

```{toctree}
:maxdepth: 2
:caption: Contents

api
```

## Installation

Requires **Python ≥ 3.10**. Install from PyPI:

```sh
pip install graphas
```

Or install from source:

```sh
git clone https://github.com/nlp-pucrs/graphas.git
cd graphas
pip install -e .
```

## Quick start

```python
import graphas

g = graphas.naiveGraph("english")
g.analyzeText("the cat sat on the mat and the cat ran")
```

See the {doc}`api` reference for the full public API, and the project
[README](https://github.com/nlp-pucrs/graphas)
for the research background and CLI usage.

## Indices

- {ref}`genindex`
- {ref}`modindex`
