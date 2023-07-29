import csv

from asr import LanguageModel, _activation_functions
from util import remove_non_speech, gen_naive_transcript, save_dat

optimal = {'Ident':     0.50,
           'Offset':    0.70,
           'Step':      0.50,
           'Sigm':      0.50,
           'Gauss':     0.70,
           'SiLU':      0.35,
           'ReLU':      0.15,
           'DReLU':     0.45,
           'spike':     1.00
           }

funcs = _activation_functions + ['spike']

params = list(set(optimal.values()))

buckets = 100

file_name = 'domain_efficacy'
file_location = '../data/test/' + file_name + '.dat'


def test(model):
    counts = [{} for _ in range(buckets + 1)]
    try:
        with open(file_location, 'r') as file:
            text = file.readline()
            skip = int(text)
            file.readline()
            for text in file:
                data = text.split(' ')
                i = int(float(data[0]) * buckets)
                for j in range(len(funcs)):
                    counts[i][funcs[j]] = float(data[j + 1])
        print('loaded row', skip, 'from', file_name+'.dat:', counts)
    except Exception as e:
        print("couldn't load from", file_name + '.dat:', e)
        skip = -1
        for bucket in range(buckets + 1):
            for func in funcs:
                counts[bucket][func] = 0.0

    with open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        score = dict()
        for i, row in enumerate(reader):
            if i <= skip:
                continue

            topics = [w for t in eval(row['topics'], {'__builtins__': None}, {}) for w in t.split()]
            print(i, topics, end=' ')
            fake_audio = model.corrupt(i, remove_non_speech(row['transcript']))
            transcript = gen_naive_transcript(model, fake_audio, preprocessed=True, verbose=False)
            domains = model.extract_domain(transcript,
                                           param=params,
                                           funcs=_activation_functions,
                                           verbose=False)

            print()
            print('par ', end=' ')
            for func in funcs:
                print(func.ljust(6), end=' ')
                if func == 'spike':
                    score['spike'] = int(model.evaluate(domains[params.index(1.0)]['ReLU'], topics) * buckets)
                else:
                    score[func] = int(model.evaluate(domains[params.index(optimal[func])][func], topics) * buckets)
            print()
            for bucket in range(buckets + 1):
                print('{:.2}'.format(bucket / buckets).ljust(4, '0'), end=' ')
                for func in funcs:
                    if score[func] == bucket:
                        counts[bucket][func] = counts[bucket][func] + 1
                    print(str(int(counts[bucket][func])).rjust(6), end=' ')
                print()
            save_dat(file_name, i, counts, ['par'] + funcs, [i / buckets for i in range(buckets + 1)])

        print('---------------DONE---------------')
        print('par ', end=' ')
        for func in funcs:
            print(func.ljust(6), end=' ')
        print()
        for bucket in range(buckets + 1):
            print('{:.2}'.format(bucket / buckets).ljust(4, '0'), end=' ')
            for func in funcs:
                print(str(counts[bucket][func]).rjust(6), end=' ')
            print()


if __name__ == '__main__':
    print('test/domain_efficacy.py')
    lm = LanguageModel()
    test(lm)
