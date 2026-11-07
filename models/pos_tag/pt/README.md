# Portuguese POS taggers

Pre-trained part-of-speech taggers for Brazilian Portuguese, to be used with
`graphas -g pos -l portuguese -p <file>` (or `posGraph("portuguese",
path_tagger=<file>)`). They are distributed with the repository only, not with
the PyPI package.

- Toolkit: NLTK
- Training corpus: [Mac-Morpho](http://nilc.icmc.usp.br/macmorpho/)
- Source: [inoueMashuu/POS-tagger-portuguese-nltk](https://github.com/inoueMashuu/POS-tagger-portuguese-nltk)
  (the source repository declares no license; the models are redistributed
  here as published by its author, with attribution)

| Tagger                   | Accuracy   | Words/s | Size    |
|--------------------------|------------|---------|---------|
| `POS_tagger_unigram.pkl` | 83.70%     | **82k** | 790 kB  |
| `POS_tagger_bigram.pkl`  | 85.18%     | 67k     | 1.37 MB |
| `POS_tagger_brill.pkl`   | **92.19%** | 30k     | 2.09 MB |

Accuracy and speed are those reported by the source repository. Load only
tagger files you trust: a pickled model runs code when it is loaded.
