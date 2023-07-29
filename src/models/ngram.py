# used reference to https://www.nltk.org/api/nltk.lm.html
from random import shuffle

import nltk
from nltk import sent_tokenize, trigrams, wordpunct_tokenize
from nltk.corpus import timit, brown, reuters, masc_tagged
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends, flatten

corpora = [timit, brown, reuters, masc_tagged]
data_path = '/home/alexb/git/fyp/data/nltk/'
n = 3
extra_words = 5


def load_corpora():
    if data_path not in nltk.data.path:
        nltk.data.path.append(data_path)

    print('1/4\tcheck nltk download')
    nltk.download(['timit', 'brown', 'reuters', 'masc_tagged', 'punkt', 'stopwords'],
                  download_dir=data_path, quiet=True)


def generate_ngram_model():
    print('2/4\textract corpora sentences')
    sentences = []
    for c in corpora:
        for s in c.sents():
            sentences.append(list(map(str.lower, s)))
    train, vocab = padded_everygram_pipeline(n, sentences)

    print('3/4\ttrain n-gram model')
    model = Laplace(n)
    model.fit(train, vocab)
    return model


def generate_ngrams(transcript):
    return list(trigrams(flatten([pad_both_ends(wordpunct_tokenize(sent), n=n) for sent in sent_tokenize(transcript)])))


def get_nrandom_words(model, n, seed=''):
    return [model.generate(random_seed=seed+str(i)) for i in range(n)]


def corrupt(model, speech, verbose=True):
    ngrams = generate_ngrams(speech)
    data = []
    markers = [int(i / 10 * len(ngrams)) for i in range(10)]
    for i, ngram in enumerate([list(n) for n in ngrams]):
        if verbose and i in markers:
            print('/', end='')
        prewords, words = ngram[:-1], ngram[-1:]
        words.extend(get_nrandom_words(model, extra_words, seed=str(ngram)))
        shuffle(words)
        data.append([prewords, words, ngram[-1]])
    return data


def test(model):
    print('counting:')
    print('count "the":           ', model.counts['the'])
    print('count "the cat":       ', model.counts[['the']]['cat'])
    print('count "the black cat": ', model.counts[['the', 'black']]['cat'])

    print('scoring:')
    print('score "the cat":       ', model.score('cat', ['the']), model.logscore('cat', ['the']))
    print('score "black cat":     ', model.score('cat', ['black']), model.logscore('cat', ['black']))
    print('score "the black cat": ', model.score('cat', ['the', 'black']), model.logscore('cat', ['the', 'black']))

    print('predictions:')
    print('gen from "the cat":       ', model.generate(5, text_seed=['the', 'cat']))
    print('gen from "the black cat": ', model.generate(5, text_seed=['the', 'black', 'cat']))
    print('gen from "":              ', model.generate(20))


if __name__ == '__main__':
    print('ngram.py')
    nltk.data.path.append(data_path)
    model = generate_ngram_model()
    test(model)
