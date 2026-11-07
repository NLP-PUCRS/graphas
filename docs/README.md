# docs

Documentation for the Graphas module. The `.md`/`.py` files here are the
**Sphinx** source; the sample texts under `examples/` are supporting material,
**not** part of the importable library.

## Structure

- **`conf.py`, `index.md`, `api.md`** — Sphinx documentation source.
- **`examples/`** — sample input texts to build graphs with (`.txt`, one text
  per file, UTF-8). Naming convention: `example_<lang>.txt` (e.g. `example_en.txt`).
  Add more languages/domains here as needed.

## Building the docs

Install the `doc` extra and run Sphinx from the repository root:

```sh
pip install -e ".[doc]"
sphinx-build -b html docs docs/_build/html
# or, from inside docs/:  make html
```

## Using an example

```sh
graphas -g naive -l english -t "$(cat docs/examples/example_en.txt)"
```
