"""Batch analysis: run a graph mode over many texts and export a feature table.

This module turns the per-text feature dictionaries of
:meth:`~graphas.textGraph.analyzeText` and
:meth:`~graphas.textGraph.analyzeWindows` into flat rows and a CSV table,
so a whole dataset (one transcription per row) can be measured in one call.
It owns no analysis logic — flattening and I/O only.
"""

import csv

from graphas.graphs.statistics import _SET_FEATURES

# nested/non-scalar keys of the analysis results that a flat table cannot hold:
# the per-window feature dictionaries and the node-SET features (their sizes
# are already reported as the scalar LCC / LSC features)
_NON_SCALAR = ('windows', *_SET_FEATURES)
# aggregate sub-dictionaries expanded into `<feature>_<aggregate>` columns
_AGGREGATES = ('mean', 'std')


def flattenStatistics(res: dict) -> dict:
    """Flatten an analysis result into a single row of scalar values.

    Accepts the dictionary of either ``analyzeText`` (whole-text) or
    ``analyzeWindows`` (windowed). The per-window dictionaries and the node
    sets are dropped, and the ``mean`` / ``std`` aggregates are expanded into
    ``<feature>_mean`` / ``<feature>_std`` columns, so the row holds scalars
    only and every text yields the same columns for the same run parameters.
    """
    row = {}
    for key, value in res.items():
        if key in _NON_SCALAR:
            continue
        if key in _AGGREGATES and isinstance(value, dict):
            row.update({f"{feat}_{key}": v for feat, v in value.items()})
        else:
            row[key] = value
    return row


def analyzeDataset(graph, texts, ids=None, window_size: int = None,
                   stride: int = 1) -> list:
    """Measure every text of a dataset and return one flat row per text.

    Runs ``graph.analyzeWindows`` on each text when ``window_size`` is given,
    ``graph.analyzeText`` otherwise, and flattens each result with
    :func:`flattenStatistics`. ``ids`` labels the rows (first column ``id``);
    it defaults to the position of the text in the dataset. A text that fails
    to analyze (e.g. too short) does NOT abort the batch: its row carries the
    ``id`` and an ``error`` message instead of features.
    """
    texts = list(texts)
    ids = list(ids) if ids is not None else list(range(len(texts)))
    if len(ids) != len(texts):
        raise ValueError("ids and texts must have the same length")
    rows = []
    for text_id, text in zip(ids, texts):
        try:
            # `is not None`: an out-of-range window_size (e.g. 0) must reach
            # slidingWindows and raise, never fall back to whole-text analysis
            if window_size is not None:
                res = graph.analyzeWindows(text, window_size=window_size,
                                           stride=stride)
            else:
                res = graph.analyzeText(text)
            rows.append({'id': text_id, **flattenStatistics(res)})
        except ValueError as e:
            rows.append({'id': text_id, 'error': str(e)})
    return rows


def writeCSV(rows: list, file) -> None:
    """Write the flat rows as a CSV table to ``file`` (path or open handle).

    The header is the union of the row keys in first-appearance order, so
    error rows and feature rows share one table (absent cells stay empty).
    """
    header = list(dict.fromkeys(key for row in rows for key in row))
    if isinstance(file, str):
        with open(file, 'w', newline='', encoding='utf-8') as handle:
            _write(rows, header, handle)
    else:
        _write(rows, header, file)


def _write(rows, header, handle):
    """Write ``rows`` under ``header`` into the open ``handle``."""
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
