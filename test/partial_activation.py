import csv

from util import save_dat
from asr import LanguageModel, _activation_functions

buckets = 20

file_name = 'partial_activation'
file_location = '../data/test/' + file_name + '.dat'


def test(model):
    score = [{} for _ in range(buckets + 1)]
    try:
        with open(file_location, 'r') as file:
            text = file.readline()
            skip = int(text)
            file.readline()
            for text in file:
                data = text.split(' ')
                i = int(float(data[0]) * buckets)
                for j in range(len(_activation_functions)):
                    score[i][_activation_functions[j]] = float(data[j + 1])
        print('loaded row', skip, 'from', file_name+'.dat:', score)
    except Exception as e:
        print("couldn't load from", file_name + '.dat:', e)
        skip = -1
        score = [{} for _ in range(buckets + 1)]

    with open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i <= skip:
                continue

            topics = [w for t in eval(row['topics'], {'__builtins__': None}, {}) for w in t.split()]
            print(i, topics)
            domains = model.extract_domain(row['transcript'],
                                           param=[i / buckets for i in range(buckets + 1)],
                                           funcs=_activation_functions,
                                           verbose=False)
            print('par ', end=' ')
            for func in _activation_functions:
                print(func.ljust(7), end=' ')
            print()
            for param in range(buckets + 1):
                print('{:.2}'.format(param / buckets).ljust(4, '0'), end=' ')
                for func in _activation_functions:
                    score[param][func] = score[param].get(func, 0) + model.evaluate(domains[param][func], topics)
                    print('{:.4}'.format(score[param][func] / i).ljust(7, '0'), end=' ')
                print()
            save_dat(file_name, i, score, ['par'] + _activation_functions, [i / buckets for i in range(buckets + 1)])

        print('---------------DONE---------------')
        print('par ', end=' ')
        for func in _activation_functions:
            print(func.ljust(7), end=' ')
        print()
        for param in range(buckets + 1):
            print('{:.2}'.format(param / buckets).ljust(4, '0'), end=' ')
            for func in _activation_functions:
                print('{:.4}'.format(score[param][func] / i).ljust(7, '0'), end=' ')
            print()


if __name__ == '__main__':
    print('test/partial_activation.py')
    lm = LanguageModel()
    test(lm)
