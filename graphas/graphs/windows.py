"""Sliding-window analysis over the continuous token flow.

This module owns the *analysis* stage of the pipeline: cutting the continuous
token flow into fixed-size windows, and summarizing the per-window feature
sequence with aggregates (mean/std) and temporal features across consecutive
windows. Graph *construction* lives in :mod:`graphas.graphs.graph` and
*measurement* in :mod:`graphas.graphs.statistics`; this module never builds
or measures a graph itself.

The sliding window is the verbosity control: T graphs of identical word
count W are built by sliding the window with stride S over the flow, so
connectedness can be compared across texts of different lengths.
"""

import numpy as np

# the node-SET features are excluded from mean/std aggregation
from graphas.graphs.statistics import _SET_FEATURES

# every temporal feature of the summary, in output order; all of them are
# ``None`` when the text yields a single window (no consecutive pair)
_TEMPORAL_FEATURES = (
    'TP', 'VP', 'VT',
    'TT', 'CV',
)


def slidingWindows(words, window_size: int = 30, stride: int = 1) -> list:
    """Cut a token flow into windows of exactly ``window_size`` tokens.

    The window advances ``stride`` tokens at a time, so every window has the same
    word count and their features are comparable (the verbosity control). A
    flow shorter than the window yields ONE window holding the whole flow.
    Works on any sequence — in the pipeline the items are ``(token, line)``
    pairs, so each window keeps the line indices for the edge rule.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2 (a graph needs an edge)")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    windows = [words[i:i + window_size]
               for i in range(0, len(words) - window_size + 1, stride)]
    return windows or [list(words)]


def window_edges(window) -> list:
    """Return the directed edges of one window of ``(token, line)`` pairs.

    Consecutive tokens are joined only when they belong to the SAME line: the
    library's segmentation rule (an edge never crosses a line break) holds
    inside the window too.
    """
    return [(u, v) for (u, li), (v, lj) in zip(window[:-1], window[1:])
            if li == lj]


class _windowStatistics():
    """Summarize the feature sequence of a windowed text.

    Takes the token windows and their feature dictionaries (index-aligned) and
    produces the per-window aggregates and the temporal features.
    """

    def __init__(self, windows, features):
        self.windows = windows
        self.features = features

    def statistics(self):
        """Return the summary dictionary of the window sequence."""
        res = {
            'number_of_windows': len(self.windows),
            'windows': self.features,
        }
        # mean/std over every SCALAR feature (the node sets are not aggregable)
        scalars = [k for k in self.features[0] if k not in _SET_FEATURES]
        for name, fn in (('mean', np.mean), ('std', np.std)):
            res[name] = {k: round(float(fn([f[k] for f in self.features])), 4)
                         for k in scalars}
        res.update(self._temporal())
        return res

    def _temporal(self):
        """Temporal features across consecutive windows.

        All of them are ``None`` when there is a single window (no consecutive
        pair to compare). Averages are over the ``T - 1`` consecutive pairs.

        - ``TP`` (transition persistence): mean Jaccard similarity of the directed
          edge sets of consecutive windows (how much structure carries over).
        - ``VP`` (vocabulary persistence): mean Jaccard similarity of the node sets
          of consecutive windows (how much vocabulary carries over).
        - ``VT`` (vocabulary turnover): mean fraction of the next window's nodes
          that are new (absent from the previous window).
        - ``TT`` (transition turnover): mean fraction of the next window's
          transitions that are new.
        - ``CV`` (connectedness volatility): mean absolute change of connectedness
          (``LSC``) between consecutive windows.
        """
        if len(self.windows) < 2:
            return dict.fromkeys(_TEMPORAL_FEATURES)
        edges = [set(window_edges(w)) for w in self.windows]
        nodes = [{word for word, _ in w} for w in self.windows]
        pairs = range(len(self.windows) - 1)
        return {
            'TP': round(float(np.mean(
                [_jaccard(edges[i], edges[i + 1]) for i in pairs])), 4),
            'VP': round(float(np.mean(
                [_jaccard(nodes[i], nodes[i + 1]) for i in pairs])), 4),
            'VT': round(float(np.mean(
                [len(nodes[i + 1] - nodes[i]) / len(nodes[i + 1])
                 for i in pairs])), 4),
            'TT': round(float(np.mean(
                [len(edges[i + 1] - edges[i]) / len(edges[i + 1])
                 if edges[i + 1] else 0.0 for i in pairs])), 4),
            'CV': round(float(np.mean(
                [abs(self.features[i + 1]['LSC'] - self.features[i]['LSC'])
                 for i in pairs])), 4),
        }



def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets; 1.0 when both are empty (identical)."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0
