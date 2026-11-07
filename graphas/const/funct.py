"""Functional collections: the supported languages and graph modes.

These sets are used to validate CLI arguments and constructor inputs early, so an
unsupported value fails fast with a clear message instead of a cryptic error later.
"""

FUNCT_LANGUAGES = {
    'arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 'german',
    'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian', 'russian',
    'spanish', 'swedish',
}
FUNCT_GRAPHS = {
    'naive', 'stem', 'pos',
}
