import csv
from random import shuffle

from asr import LanguageModel
from util import save_dat, remove_non_speech, update_totals, gen_naive_transcript

param = 1.0
func = 'Step'

file_name = 'true_adapt_' + str(int(param*100)) + '_' + func
file_location = '../data/test/' + file_name + '.dat'
extra_words = 2


def test(model):
    try:
        with open(file_location, 'r') as file:
            text = file.readline()
            skip = int(text)
            file.readline()
            text = file.readline().split(' ')
            totals = [int(n) for n in text]
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
            speech = remove_non_speech(raw)
            fake_audio = model.corrupt(i, speech)  # generate ngrams
            naive_transcript = gen_naive_transcript(model, fake_audio, preprocessed=True, verbose=False)
            domain = model.extract_domain(naive_transcript, verbose=False, funcs=[func], param=[param])
            print('domain:', domain, row['topics'])
            naive_transcript = naive_transcript.split(' ')
            print('t=i\t\t\t\tt=a\t\t\t\tt=i=a\t\t\t Improvement\t', ['true', 'invar', 'adapt'])
            for j, data in enumerate(fake_audio):
                invar = naive_transcript[j+2]
                adapt = model.predict_word(data[0], data[1], domain, verbose=False)
                totals = update_totals(totals, data[2], invar, adapt)
            save_dat(file_name, i, data=totals, columns=['count', 'invar', 'adapt', 'both'])


if __name__ == '__main__':
    print('test/true_adaptive.py')
    lm = LanguageModel()
    test(lm)
