# `/test`

### In this directory:
| File(s)                                       |                   | Description                                                                                |
|-----------------------------------------------|-------------------|--------------------------------------------------------------------------------------------|
| `README.md`                                   |                   | The file you're reading right now, a summary of the contents of this folder                |                                                                                                                                 |
| `partial_adapt.py`, `true_adapt.py`           | **Core Test**     | Compare accuracy of `predicted_word` both with and without domain, on (unseen) TED dataset |                                                                                                                                |
| `partial_activation.py`, `true_activation.py` | **Training Test** | Tests efficacy of different activation functions in LM across a range of parameters        |
| `domain_efficacy.py`                          | **Training Test** | Tests the efficacy of `extract_domain(text)`                                               |
| `word_examples.py`                            | Examples          | Examples of `predict_word(prewords, word, domain='')` both with and without domain         |
| `speed.py`                                    | **Core Test**     | Times how fast LanguageModel takes to transcribe                                           |
| `perplexity.py`                               | Follow up Test    | Calculate Perplexity of Language Model                                                     |