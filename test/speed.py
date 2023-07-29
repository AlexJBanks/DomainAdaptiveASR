import csv
import time
from random import shuffle

from asr import LanguageModel
from util import save_dat, remove_non_speech, update_totals, gen_naive_transcript

file_name = 'speed'
file_location = '../data/test/' + file_name + '.dat'
extra_words = 2


def test(model):
    try:
        with open(file_location, 'r') as file:
            text = file.readlines()[-1]
        splits = text.split(' ')
        skip = int(splits[0])
        print('loaded row', skip, 'from', file_name + '.dat')
    except Exception as e:
        print("couldn't load from " + file_name + '.dat:', e)
        skip = -1

    with open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i <= skip:
                continue
            print(i, end=' ')

            raw = row['transcript']                 # Preprocessing
            speech = remove_non_speech(raw)         #
            fake_audio = model.corrupt(i, speech)   #
            print('.', end='\t')                    #

            start = time.time()                                                                                 # ---- #
            naive_transcript = gen_naive_transcript(model, fake_audio, preprocessed=True, verbose=False)               #
            naive_done = time.time()                                                                            # ---- #
            domain = model.extract_domain(naive_transcript, verbose=False)                                             #
            extract_done = time.time()                                                                          # ---- #
            for j, data in enumerate(fake_audio):                                                                      #
                model.predict_word(data[0], data[1], domain, verbose=False)                                            #
            end = time.time()                                                                                   # ---- #

            naive = naive_done - start
            extract = extract_done - naive_done
            adapt = end - extract_done
            total = end - start
            length = row['duration']
            print('\tt_time:', int(total), '\tv_time:', length,
                  '\tnaive:', '{:.2%}'.format(naive/total).rjust(7),
                  '\textract:', '{:.2%}'.format(extract/total).rjust(7),
                  '\tadapt:', '{:.2%}'.format(adapt/total).rjust(7),
                  '\tratio:', '{:.4}'.format(total/int(length)).ljust(8, '0'),
                  '\tdomain:', domain)
            save_dat(file_name, i, data=[str(i), length, str(naive), str(extract), str(adapt), str(total)],
                     columns=['i', 'video', 'naive', 'extract', 'adapt', 'total'], append=True)


if __name__ == '__main__':
    print('test/speed.py')
    lm = LanguageModel()
    lm.predict_word([], [], "") # force load w2v
    test(lm)
