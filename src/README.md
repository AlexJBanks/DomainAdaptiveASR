# `/src`

### In this directory:
| File(s)          | Description                                                                                                                                        |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `README.md`      | The file you're reading right now, a summary of the contents of this folder                                                                        |
| `asr.py`         | `LanguageModel()`, for preloading models, which is used to extract domain and predict words                                                        |
| `gen_corrupt.py` | Script that pre-generates corrupted TED Talk test data. Can be done on the fly, but this will speed up execution without running tests themselves. |
| `ted_scraper.py` | Script that downloads all needed data from TED.org website. Requires `/data/TED/ted_talks_en.csv` has already been retrieved.                      |
| `util.py`        | Various helper functions                                                                                                                           |
| `/models`        | Contains NLTK corpora and model scripts needed for `LanguageModel()`                                                                               |


### `ted_scraper.py`
 - `ted_talks_en.csv` should already be loaded before running `ted_scraper.py`
 - additionally, we're assuming ted-talks are sufficiently shuffled before processing.
 - `ted_scraper.py` can be ran with argument `storage=/dir/` where dir is an alternative location for data to be stored in.
This is useful if you don't want to store all data within `/data`
