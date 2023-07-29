import csv

from asr import LanguageModel
from util import save_dat, remove_non_speech, update_totals, gen_naive_transcript

file_name = 'perplexity'
file_location = '../data/test/' + file_name + '.dat'


def test(model):
    try:
        with open(file_location, 'r') as file:
            text = file.readline()
            skip = int(text)
            file.readline()
            text = file.readline().split(' ')
            count = int(text[0])
            mult = int(text[1])
        print('loaded row', skip, 'from', file_name + '.dat:', count, mult)
    except Exception as e:
        print("couldn't load from " + file_name + '.dat:', e)
        skip = -1
        count = 0
        mult = 1

    with open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        print('i\tcount\tPP')
        for i, row in enumerate(reader):
            if i <= skip:
                continue

            print(i, end='\t')
            fake_audio = model.corrupt(i, remove_non_speech(row['transcript']), verbose=False)  # generate ngrams

            if not len(fake_audio):
                continue

            for data in fake_audio:
                count = count + 1
                this = model.predict_word(data[0], data[2], verbose=False, probability=True)
                mult = (mult ** ((count - 1) / count)) * (this ** (-1 / count))
            print(count, '\t', mult)
            save_dat(file_name, i, data=[count, mult], columns=['count', 'mult'])
        print('\n--finished--\n\t', count, '\t', mult)


if __name__ == '__main__':
    print('test/perplexity.py')
    lm = LanguageModel()
    test(lm)
