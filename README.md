# Domain Adaptive Automatic Speech Recognition
### Final Year Project code submission for MSci Comp Sci degree

This repo contains the code contribution side of my FYP (Dissertation) in 2022.
It complements the 13,000 word report, not included.

## Brief Summary
ASR can be inaccurate.
Some of those inaccuracies come about when the ASR transcriber runs into ambiguities and doesn't understand the context to disambiguate.
If we got the transcriber to understand the Domain (Topic of Speech) it should be more accurate.
This project included a series of experiments with methodologies to improve the accuracy of transcription by making the ASR transcriber include knowledge of domain.


The project was submitted in June 2022, and received a 1st.
It's just taken me until July 2023 to actually upload it.

original file as follows
___

# Domain Adaptive Automatic Speech Recognition
### Alex Banks - 1780069
#### M60 Computer Science Project
___

Welcome.
You wanna run this code? You sure? ok.

## Setup
1. Make sure you have all the required libraries installed
   - Full list of requirements in `requirements.txt`
2. Gather TED Ultimate Dataset from [kaggle.com/miguelcorraljr/ted-ultimate-dataset](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset)
   - You only need the English spoken talks `ted_talks_en.csv`
   - Place this in `/data/TED/`

## Directories
Each subdirectory should contain a `README.md` containing information on what is contained in each folder.
Below is the contents for this directory:

### In this directory:
| File(s)            | Description                                                                              |
|--------------------|------------------------------------------------------------------------------------------|
| `README.md`        | The file you're reading right now, a summary of the contents of this folder              |
| `requirements.txt` | List of requirements...                                                                  |
| `/src`             | Directory containing all the scripts that make up the core of the project                |
| `/data`            | Directory containing data, models, and other downloaded utilities needed for the project |
| `/test`            | Directory containing test scripts for analysis and evaluation of DA-ASR                  |
