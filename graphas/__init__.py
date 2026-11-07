"""Graphas: speech graph analysis.

Build a directed word graph (``networkx.MultiDiGraph``) from a speech
transcription and extract the structural features used to quantify how organized
or fragmented a discourse is. The method is language-agnostic and needs no
linguistic pre-processing or reference corpus.

Public API
----------
``naiveGraph`` / ``stemGraph`` / ``posGraph``
    The three graph modes (node representations) over a common base.
``textGraph``
    Abstract base class of the three modes.
``ensure_nltk_data``
    Pre-download the NLTK data the library needs (otherwise fetched lazily on the
    first graph construction).
``analyzeDataset`` / ``flattenStatistics`` / ``writeCSV``
    Batch analysis: run a graph mode over many texts, flatten each result into
    one row of scalars, and export the feature table as CSV.

Example
-------
>>> import graphas
>>> g = graphas.naiveGraph("english")
>>> g.analyzeText("the cat sat on the mat and the cat ran")["LSC"]
6
"""

from graphas.batch import analyzeDataset, flattenStatistics, writeCSV
from graphas.const.app import APP_VERSION as __version__
from graphas.graphs.graph import (
    ensure_nltk_data,
    naiveGraph,
    posGraph,
    stemGraph,
    textGraph,
)

__all__ = [
    "naiveGraph",
    "stemGraph",
    "posGraph",
    "textGraph",
    "ensure_nltk_data",
    "analyzeDataset",
    "flattenStatistics",
    "writeCSV",
    "__version__",
]
