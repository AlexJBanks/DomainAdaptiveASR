import csv
from random import shuffle

from asr import LanguageModel
from util import save_dat, remove_non_speech, update_totals

file_name = 'partial_adapt'
file_location = '../data/test/' + file_name + '.dat'


def test(model):
    try:
        with open(file_location, 'r') as file:
            text = file.readline()
            skip = int(text)
            file.readline()
            text = file.readline().split(' ')
            totals = [int(n) for n in text[:-1]]
        print('loaded row', skip, 'from', file_name + '.dat:', totals)
    except Exception as e:
        print("couldn't load from " + file_name + '.dat:', e)
        skip = -1
        totals = [0, 0, 0, 0]

    with open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i <= skip:
                continue
            print(i, end=' ')
            raw = row['transcript']
            domain = model.extract_domain(raw, verbose=False)
            print('domain:', domain, row['topics'])
            ngrams = model.generate_ngrams(remove_non_speech(raw))  # generate ngrams
            print('t=i\t\t\t\tt=a\t\t\t\tt=i=a\t\t\t Improvement\t', ['true', 'invar', 'adapt'])
            for ngram in [list(n) for n in ngrams]:
                prewords, words = ngram[:-1], ngram[-1:]
                words.extend(model.get_nrandom_words(4, seed=str(ngram)))
                shuffle(words)
                invar = model.predict_word(prewords, words, verbose=False)
                adapt = model.predict_word(prewords, words, domain, verbose=False)
                totals = update_totals(totals, ngram[-1], invar, adapt)
            save_dat(file_name, i, data=totals, columns=['count', 'invar', 'adapt', 'both'])


if __name__ == '__main__':
    print('test/partial_adapt.py')
    lm = LanguageModel()
    test(lm)
