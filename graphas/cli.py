"""Command-line interface for Graphas.

Exposes the ``graphas`` console command (and ``python -m graphas``) that
builds a graph from one or more texts and prints the feature dictionary of each.
"""

import argparse
import csv
import sys

import graphas.const as const
import graphas.graphs as graph
from graphas.batch import analyzeDataset, writeCSV

__author__ = const.APP_AUTHORSHIP
__version__ = const.APP_VERSION


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``graphas`` command."""
    parser = argparse.ArgumentParser(
        prog="graphas",
        description=const.APP_DESCRIPTION,
        epilog=f"Authors: {__author__}",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-t", action="extend", nargs="+", dest="text", default=list(),
        help="text used to generate a graph")
    source.add_argument(
        "--csv", dest="csv", default=None, metavar="PATH",
        help="batch mode: CSV dataset with one transcription per row "
             "(see --text-column / --id-column)")
    parser.add_argument(
        "--text-column", dest="text_column", default="text", metavar="NAME",
        help="name of the transcription column of the --csv dataset. "
             "default: %(default)s")
    parser.add_argument(
        "--id-column", dest="id_column", default=None, metavar="NAME",
        help="name of the id column of the --csv dataset. "
             "default: the row position")
    parser.add_argument(
        "-o", "--output", dest="output", default=None, metavar="PATH",
        help="write the results as a CSV feature table (one row per text) "
             "instead of printing dictionaries")
    parser.add_argument(
        "-g", default="naive", dest="graph",
        choices=sorted(const.FUNCT_GRAPHS),
        help="select the type of graph. default: %(default)s")
    parser.add_argument(
        "-l", default="portuguese", dest="lang",
        choices=sorted(const.FUNCT_LANGUAGES),
        help="language for nltk tokenizers. default: %(default)s")
    parser.add_argument(
        "-p", action="store", dest="pos_tagger", default=None,
        help="path to a custom pickled POS tagger (load only files you trust: "
             "unpickling runs arbitrary code)")
    parser.add_argument(
        "-s", "--stopwords", action="store_true", help="skip stopwords")
    parser.add_argument(
        "-w", "--window", type=int, default=None, dest="window",
        help="sliding-window analysis: window size in words (e.g. 30). "
             "omit for whole-text analysis")
    parser.add_argument(
        "--stride", type=int, default=1, dest="stride",
        help="words the sliding window advances at a time (with -w). "
             "default: %(default)s")
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"%(prog)s (version {__version__})")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the analysis described by the parsed ``args`` and exit."""
    # Build the requested graph mode. Construction can fail on an invalid language
    # or a missing custom tagger; report those cleanly instead of dumping a
    # traceback at the user.
    try:
        match args.graph:
            case "naive":
                gr = graph.naiveGraph(language=args.lang, stopwords=args.stopwords)
            case "stem":
                gr = graph.stemGraph(language=args.lang, stopwords=args.stopwords)
            case "pos":
                gr = graph.posGraph(language=args.lang, path_tagger=args.pos_tagger)
    except (ValueError, FileNotFoundError) as e:
        print(f"could not build the '{args.graph}' graph: {e}", file=sys.stderr)
        sys.exit(1)

    # Collect the texts and their ids: from the CSV dataset in batch mode,
    # from the -t arguments otherwise (ids default to the position).
    if args.csv:
        try:
            texts, ids = _read_dataset(args.csv, args.text_column,
                                       args.id_column)
        except (OSError, KeyError) as e:
            print(f"could not read the dataset '{args.csv}': {e}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        texts, ids = args.text, None

    if args.output or args.csv:
        # table mode: one flat row per text, written as CSV (to the --output
        # path, or to stdout when analyzing a --csv dataset without one).
        rows = analyzeDataset(gr, texts, ids=ids, window_size=args.window,
                              stride=args.stride)
        writeCSV(rows, args.output if args.output else sys.stdout)
        for row in rows:
            if 'error' in row:
                print(f"error processing text '{row['id']}': {row['error']}",
                      file=sys.stderr)
        sys.exit(1 if any('error' in row for row in rows) else 0)

    # print mode (-t without -o): the full stats dictionary of each text,
    # windowed when -w is given.
    exit_code = 0
    for text in texts:
        try:
            # `is not None`: an out-of-range window size (e.g. -w 0) must
            # reach the windowed analysis and raise, never fall back silently
            if args.window is not None:
                print(gr.analyzeWindows(text, window_size=args.window,
                                        stride=args.stride))
            else:
                print(gr.analyzeText(text))
        except Exception as e:
            # report on stderr and keep going, so one bad text does not abort
            # the rest of the batch; exit non-zero if any text failed.
            print(f"error processing text '{text}': {e}", file=sys.stderr)
            exit_code = 1
    sys.exit(exit_code)


def _read_dataset(path: str, text_column: str, id_column: str):
    """Read the (texts, ids) of a CSV dataset; ids is None without an id column.

    Raises
    ------
    KeyError
        If ``text_column`` or ``id_column`` is not a column of the CSV.
    """
    # utf-8-sig: strip the UTF-8 BOM that Excel-exported CSVs carry (it would
    # otherwise corrupt the first column name); reads BOM-less files unchanged.
    with open(path, newline='', encoding='utf-8-sig') as handle:
        records = list(csv.DictReader(handle))
    texts = [record[text_column] for record in records]
    ids = [record[id_column] for record in records] if id_column else None
    return texts, ids


def run(argv=None) -> None:
    """Entry point for the ``graphas`` console script."""
    main(build_parser().parse_args(argv))


if __name__ == "__main__":
    run()
