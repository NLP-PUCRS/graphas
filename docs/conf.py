"""Sphinx configuration for the Graphas documentation.

Build locally with the ``doc`` extra installed::

    pip install -e ".[doc]"
    sphinx-build -b html docs docs/_build/html
"""

from graphas import __version__

# -- Project information ------------------------------------------------------
project = "Graphas"
author = "Henry Saenz, Kelly Beltran, Carlos Gomes and Facundo Carrillo"
copyright = (  # noqa: A001
    "2026, Natural Language Processing Group from Pontifícia Universidade "
    "Católica do Rio Grande do Sul (PUCRS), Brazil"
)
release = __version__
version = __version__

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
]

autosummary_generate = True
numpydoc_show_class_members = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "nltk": ("https://www.nltk.org/", None),
}

exclude_patterns = ["_build", "README.md", "examples/*.txt", "Thumbs.db", ".DS_Store"]

# -- HTML output --------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"Graphas {release}"
