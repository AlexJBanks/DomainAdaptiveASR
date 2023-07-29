# `/src/models`

### In this directory:
| File(s)       | Description                                                                                   |
|---------------|-----------------------------------------------------------------------------------------------|
| `README.md`   | The file you're reading right now, a summary of the contents of this folder                   |
| `ngram.py`    | Returns ngram model for selected NLTK corpora. Stores raw data in `data/nltk`                 |
| `word2vec.py` | Returns word2vec model based on Google News vectorisation. Stores raw data in `data/word2vec` |

## Corpora:
- masc: spoken language corpus from modern American English sources
- reuters: written language corpus from Reuters articles from 21st century
- timit: spoken language corpus from english literature, specifically aimed for ASR use-cases.
- brown: written and spoken language from a wide range of sources, aiming to give a solid foundation.

## Word2Vec source:
- Google News 