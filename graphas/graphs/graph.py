"""Core text-to-graph modes and graph feature extraction.

This module owns graph *construction*: turning a text into a directed word
graph. It exposes three node representations
(``naiveGraph`` / ``stemGraph`` / ``posGraph``) over a common base
(``textGraph``). The *measurement* stage (the feature extractor
``_graphStatistics``) lives in :mod:`graphas.graphs.statistics`.

Segmentation rule:
    The word flow is split on **line breaks only** (``str.splitlines``).
    Punctuation never breaks the flow: ``cleanText`` strips it and the words on
    either side stay consecutive, so an edge is still drawn across a period or a
    comma. The only separator is a newline in the source text — matching the way
    a continuous speech transcription is laid out (one utterance per line). POS
    mode still sentence-tokenizes each line internally for tagging accuracy, but
    concatenates the tags so its edges stay continuous across sentences too.

Design note:
    ``textGraph`` is a two-stage pipeline. Each mode only implements
    ``_line2words`` (how ONE line becomes normalized tokens); the base class
    turns that into a per-line token structure (``_text2words``) and then into
    the whole-text graph (``_text2graph``). The whole-text graph cuts edges at
    line boundaries; the sliding-window mode flattens ``_text2words`` into a
    single continuous flow without touching the modes.
"""

import re
from abc import ABC, abstractmethod
from os import path

import joblib
import networkx as nx

# nltk.pos_tag supports only eng and rus
from nltk import pos_tag

# imported under an alias so it does not clash with the `stopwords` attribute
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from unidecode import unidecode

# supported languages, used to validate the `language` argument early
from graphas.const.funct import FUNCT_LANGUAGES

# the feature extractor and the NLTK bootstrap each live in their own module
# (measurement and infrastructure are separate from construction); re-imported
# here so `analyzeText`, and importers of
# `graphas.graphs.graph._graphStatistics` / `.ensure_nltk_data`, keep working.
from graphas.graphs.nltk_data import ensure_nltk_data, ensure_resource
from graphas.graphs.statistics import _graphStatistics
from graphas.graphs.windows import (
    _windowStatistics,
    slidingWindows,
    window_edges,
)

__all__ = ["textGraph", "naiveGraph", "stemGraph", "posGraph", "ensure_nltk_data"]

# the one message both public analyze methods raise on a too-short text
# (a graph needs at least one edge, i.e. two tokens)
_TOO_SHORT_ERROR = "text has fewer than 2 tokens after tokenization"


def _sentences(line, language):
    """Split one line into sentences with NLTK's punkt model for ``language``.

    Punkt ships models for only part of the supported languages; for the others
    (e.g. Arabic, Hungarian, Romanian) the English model is used, which splits
    on the same terminal punctuation.
    """
    try:
        return sent_tokenize(line, language=language)
    except LookupError:
        return sent_tokenize(line, language="english")


class textGraph(ABC):
    """Abstract base of the graph modes: a two-stage text-to-graph pipeline.

    Subclasses implement :meth:`_line2words` (how one line becomes normalized
    tokens); the base turns that into the whole-text graph and its feature
    dictionary via :meth:`analyzeText`.
    """

    def __init__(self, language: str, stopwords: bool = False):
        # Fetch NLTK data lazily on first construction (never at import time), and
        # before building the stopword cache, which reads the stopwords corpus.
        ensure_nltk_data()
        # `language` is validated in the setter; do it before building the
        # stopword cache so an invalid language fails fast at construction.
        self.language = language
        self.stopwords = stopwords
        # build the stopword set ONCE (O(1) membership on every filter_words()
        # call). Empty set when stopword removal is disabled.
        self._stopset = (set(nltk_stopwords.words(self.language))
                         if self.stopwords else set())

    @property
    def language(self) -> str:
        """The validated, lower-cased language name."""
        return self.__language

    @language.setter
    def language(self, language: str):
        lang = language.lower()
        # validate against the supported set and raise a clear, actionable
        # error at construction, instead of a cryptic NLTK LookupError later.
        if lang not in FUNCT_LANGUAGES:
            raise ValueError(
                f"unsupported language '{language}'; "
                f"choose one of {sorted(FUNCT_LANGUAGES)}")
        self.__language = lang

    @property
    def stopwords(self) -> bool:
        """Whether stopwords are removed when building the graph."""
        return self.__stopwords

    @stopwords.setter
    def stopwords(self, stopwords: bool):
        self.__stopwords = bool(stopwords)

    @abstractmethod
    def _line2words(self, line: str) -> list:
        """Convert one line into this mode's normalized token sequence.

        Words / stems / POS tags depending on the mode. Implemented per subclass.
        """
        ...

    def _text2words(self, text: str) -> list:
        """Tokenize the whole text into a list of lines of normalized tokens.

        The text is split on line breaks only (``str.splitlines``); punctuation
        inside a line does not split the flow. Splitting happens BEFORE cleaning,
        because ``cleanText`` would otherwise turn newlines into spaces and erase
        the only separator. One tokenization, reusable by both the whole-text
        graph and the continuous-flow sliding-window mode.
        """
        return [self._line2words(line) for line in text.splitlines()]

    def _text2flow(self, text: str) -> list:
        """Flatten the per-line tokenization into one flow of (token, line) pairs.

        The sliding-window mode slides over this flow — the window RANGES over
        the whole transcription — but each token keeps its line index so the
        edge rule can still cut at line breaks inside a window. Reuses the same
        tokenization as :meth:`_text2words`; the whole-text :meth:`_text2graph`
        is untouched.
        """
        return [(word, i)
                for i, line in enumerate(self._text2words(text))
                for word in line]

    @staticmethod
    def _pairs(words):
        """Turn consecutive tokens into directed pairs (the edge rule).

        The single place that defines "consecutive words -> directed edge",
        shared by the whole-text builder (:meth:`_text2graph`) and the
        single-sequence builder (:meth:`_words2graph`). ``words`` is a real list,
        so ``words[:-1]`` and ``words[1:]`` are independent slices -> every
        consecutive pair is emitted exactly once.
        """
        return zip(words[:-1], words[1:])

    def _lines2graph(self, lines):
        """Assemble the whole-text graph from the per-line tokenization.

        Edges never cross line breaks (each line is assembled independently); a
        period or comma inside a line keeps the words connected.
        """
        gr = nx.MultiDiGraph()
        for words in lines:
            gr.add_edges_from(self._pairs(words))
        return gr

    def _text2graph(self, text):
        """Build the whole-text graph (tokenization + per-line assembly)."""
        return self._lines2graph(self._text2words(text))

    @staticmethod
    def _words2graph(window):
        """Assemble a directed multigraph from ONE window of (token, line) pairs.

        This is the construction primitive the sliding-window mode uses per
        window. The edge rule is the same as the whole-text graph: consecutive
        tokens are joined only when they belong to the SAME line (an edge never
        crosses a line break, even inside a window). Unlike :meth:`_text2graph`
        it adds the nodes explicitly, so a degenerate window (0 or 1 edge)
        still yields a well-formed graph instead of an empty one.
        """
        gr = nx.MultiDiGraph()
        gr.add_nodes_from(word for word, _ in window)
        gr.add_edges_from(window_edges(window))
        return gr

    def analyzeText(self, text):
        """Build the graph for ``text`` and return its feature dictionary.

        The dictionary is led by ``K``, the token count of the preprocessed
        text (so ``N <= K`` always holds and ``N/K`` measures lexical
        diversity), followed by every structural feature of the graph.

        Raises
        ------
        ValueError
            If the text has fewer than 2 tokens after tokenization.
        """
        # tokenize once: the same per-line tokenization yields both the token
        # count K (a property of the flow, unrecoverable from the graph) and
        # the graph itself
        lines = self._text2words(text)
        dgr = self._lines2graph(lines)
        # fail fast at the public boundary with a clear message, instead of
        # crashing later inside _graphStatistics (max() over empty components).
        if dgr.number_of_nodes() < 2:
            raise ValueError(_TOO_SHORT_ERROR)
        res = {'K': sum(len(words) for words in lines)}
        res.update(_graphStatistics(dgr).statistics())
        return res

    def analyzeWindows(self, text, window_size: int = 30, stride: int = 1):
        """Sliding-window analysis of ``text`` (the verbosity control).

        Slides a window of ``window_size`` tokens with stride ``stride`` over the
        token flow of the whole transcription (the window RANGES across line
        breaks, but the edge rule is the same as :meth:`analyzeText`: a line
        break still cuts the edge inside a window), builds one graph per
        window, and
        measures each with the same feature extractor as the whole-text
        analysis. Returns a dictionary with the run parameters (including
        ``K``, the token count of the flow), the per-window feature
        dictionaries (``windows``), their aggregates (``mean`` / ``std``),
        and the temporal features across consecutive windows
        (``TP``, ``VP``, ``VT``, ``TT``, ``CV`` — all ``None`` when there
        is a single window). A text shorter than
        the window yields one window holding the whole flow.

        Raises
        ------
        ValueError
            If the text has fewer than 2 tokens after tokenization, or if
            ``window_size`` / ``stride`` are out of range.
        """
        flow = self._text2flow(text)
        if len(flow) < 2:
            raise ValueError(_TOO_SHORT_ERROR)
        windows = slidingWindows(flow, window_size=window_size, stride=stride)
        features = [_graphStatistics(self._words2graph(w)).statistics()
                    for w in windows]
        res = {'window_size': window_size, 'stride': stride, 'K': len(flow)}
        res.update(_windowStatistics(windows, features).statistics())
        return res

    def filter_words(self, word: str):
        """Return ``False`` if the word is empty or, when enabled, a stopword.

        Otherwise return ``True``.
        """
        if not word:
            return False
        # O(1) set lookup against the cached stopword set
        if self.stopwords and word in self._stopset:
            return False
        return True

    @staticmethod
    def cleanText(text, no_accents=False):
        """Lower-case, strip, and replace non-word characters with spaces.

        With ``no_accents=True`` accents are also removed (via ``unidecode``).
        """
        base = unidecode(text.lower().strip()) if no_accents else text.lower().strip()
        # raw string so `\w` is a literal regex escape and Python emits no
        # SyntaxWarning (invalid escape sequence).
        return re.sub(r"[^\w ]+", " ", base)


class naiveGraph(textGraph):
    """Graph mode where each distinct word becomes a node (no transformation)."""

    def __init__(self, language: str, stopwords: bool = False):
        super().__init__(language=language, stopwords=stopwords)

    def _line2words(self, line):
        cleaned = self.cleanText(line)
        return [w for w in word_tokenize(cleaned, preserve_line=True)
                if self.filter_words(w)]


class stemGraph(textGraph):
    """Graph mode where words are stemmed, collapsing surface variants to one node."""

    def __init__(self, language: str, stopwords: bool = False):
        super().__init__(language=language, stopwords=stopwords)
        # build the stemmer ONCE and reuse it on every _text2graph call.
        self._stemmer = SnowballStemmer(self.language)

    def _line2words(self, line):
        cleaned = self.cleanText(line)
        return [self._stemmer.stem(w)
                for w in word_tokenize(cleaned, preserve_line=True)
                if self.filter_words(w)]


class posGraph(textGraph):
    """Graph mode where each word is replaced by its part-of-speech tag."""

    def __init__(self, language: str, path_tagger: str = None):
        # posGraph does NOT filter stopwords: the tagger needs the full token
        # sequence, so stopword removal stays disabled (base default).
        super().__init__(language=language)
        self.pos_tagger = path_tagger
        # use the validated, lower-cased language of the base class
        self.pos_lang = self.language

    @property
    def pos_lang(self) -> str:
        """NLTK language code for the built-in tagger (``eng`` / ``rus``)."""
        return self.__pos_lang

    @pos_lang.setter
    def pos_lang(self, lang: str):
        self.__pos_lang = None
        match lang:
            case 'english':
                # model is available by default
                self.__pos_lang = 'eng'
            case 'russian':
                # ensure the Russian tagger model is available (the resource
                # name NLTK >= 3.9 loads is `averaged_perceptron_tagger_rus`)
                ensure_resource('taggers/averaged_perceptron_tagger_rus',
                                'averaged_perceptron_tagger_rus')
                self.__pos_lang = 'rus'
            case _:
                if not self.pos_tagger:
                    raise ValueError(
                        f"'{lang}' is not supported by the default nltk pos tagger "
                        f"and no custom model was loaded")

    @property
    def pos_tagger(self):
        """The loaded custom tagger's ``tag`` callable, or ``None``."""
        return self.__pos_tagger

    @pos_tagger.setter
    def pos_tagger(self, path_tagger: str):
        self.__pos_tagger = None
        if path_tagger:
            # A path was given: fail loudly if it is missing, instead of silently
            # falling back to the default tagger and later raising a misleading
            # "no custom model was loaded" error.
            if not path.isfile(path_tagger):
                raise FileNotFoundError(
                    f"custom POS tagger not found: '{path_tagger}'")
            # assume a pickled model file exposing a `.tag` method
            self.__pos_tagger = joblib.load(path_tagger).tag

    def _line2words(self, line):
        # Tag each sentence of the line separately (the tagger is more accurate
        # with sentence context), but CONCATENATE the tag sequences so the edges
        # stay continuous across sentence boundaries: only a line break separates
        # the flow, a period does not.
        tags = []
        for sentence in _sentences(line, self.language):
            cleaned = self.cleanText(sentence)
            tokens = word_tokenize(cleaned, preserve_line=True)
            if self.pos_tagger:                   # custom pickled tagger
                tagged = self.pos_tagger(tokens)
            else:                                 # nltk built-in (eng / rus)
                tagged = pos_tag(tokens, lang=self.pos_lang)
            # extend a plain LIST of POS tags. The base class zips two
            # independent slices of it, so consecutive pairs are built correctly.
            tags.extend(t[1] for t in tagged)
        return tags
