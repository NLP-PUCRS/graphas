"""Lazy NLTK data bootstrap.

Downloading the NLTK data the library needs is infrastructure, not graph
construction, so it lives in its own module. The rule it enforces: merely
*importing* the library must never trigger a download; the data is fetched on the
first graph construction (or when a caller invokes :func:`ensure_nltk_data`
explicitly), and only the resources that are actually missing are downloaded.
"""

import nltk
from nltk.downloader import download as nltk_download

# The resources every graph mode may need: (path checked with ``nltk.data.find``,
# package id passed to the downloader). ``punkt_tab`` and
# ``averaged_perceptron_tagger_eng`` are the resource names NLTK >= 3.9 loads.
_REQUIRED = (
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
)

# Guard so the availability check runs at most once per process.
_NLTK_BOOTSTRAPPED = False


def _is_available(resource: str) -> bool:
    """Return whether an NLTK resource is already present on this machine."""
    try:
        nltk.data.find(resource)
    except LookupError:
        return False
    return True


def ensure_resource(resource: str, package: str, quiet=True):
    """Download one NLTK ``package`` unless its ``resource`` path is present.

    Parameters
    ----------
    resource : str
        Path checked with ``nltk.data.find`` (e.g. ``"taggers/x"``).
    package : str
        Package id passed to the NLTK downloader when the resource is missing.
    quiet : bool, default True
        Passed to the NLTK downloader.
    """
    if not _is_available(resource):
        nltk_download(package, quiet=quiet)


def ensure_nltk_data(packages=None, quiet=True):
    """Download the NLTK data this library needs, lazily.

    Merely *importing* the library must never trigger a download; instead the
    data is fetched on the first graph construction (or when a caller invokes this
    function explicitly). Only missing resources are downloaded, so a machine that
    already has them never touches the network. Safe to call repeatedly.

    Parameters
    ----------
    packages : iterable of str, optional
        NLTK package ids to download unconditionally, instead of the default
        set the library needs (``punkt_tab``, ``stopwords``,
        ``averaged_perceptron_tagger_eng``).
    quiet : bool, default True
        Passed to the NLTK downloader.
    """
    global _NLTK_BOOTSTRAPPED
    if packages is not None:
        for package in packages:
            nltk_download(package, quiet=quiet)
        return
    if _NLTK_BOOTSTRAPPED:
        return
    for resource, package in _REQUIRED:
        ensure_resource(resource, package, quiet=quiet)
    _NLTK_BOOTSTRAPPED = True
