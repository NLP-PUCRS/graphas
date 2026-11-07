---
name: Bug report
about: Report a bug or unexpected behavior in Graphas
title: "[BUG] "
labels: bug
assignees: ''
---

## Describe the bug

<!-- A concise description of the incorrect or unexpected behavior. -->

## Minimal reproducible example

<!-- The shortest input that triggers the problem. Include BOTH how you called
it (Python or CLI) AND the exact input text, since results here depend on the
text. -->

**Graph mode:** <!-- naive / stem / pos -->
**Language (`-l`):** <!-- e.g. english, portuguese -->
**Stopwords / custom POS tagger:** <!-- on/off ; path if any -->

```python
# Python API
import graphas
g = graphas.naiveGraph("english")
g.analyzeText("...the exact input text...")
```

```sh
# or the CLI command you ran
graphas -g naive -l english -t "...the exact input text..."
```

## Expected behavior

<!-- What you expected to happen. -->

## Actual behavior

<!-- What happened instead. Paste the FULL error traceback if there was one. -->

## Environment

- Operating system:
- Python version: <!-- `python --version` -->
- Graphas version: <!-- `pip show graphas` -->

## Additional context

<!-- Anything else that might help (NLTK data issues, locale, etc.). -->
