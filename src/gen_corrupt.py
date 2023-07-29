import csv
import os.path

from asr import LanguageModel
from util import remove_non_speech


def locker():
    print('--locker--')
    with open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):

            file = '../data/corrupt/' + str(i).rjust(4, '0')
            if os.path.exists(file+'.bin'):
                open(file + '.lock', 'a').close()
            if not i % 100:
                print(i)


def gen(lm, nice=True):
    print('--nice:' + str(nice) + '--')
    with open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):

            file = '../data/corrupt/' + str(i).rjust(4, '0')
            if os.path.exists(file+'.test') or os.path.exists(file+'.bin') or (os.path.exists(file+'.lock') and nice):
                continue
            print(i, end=' ')
            lm.corrupt(i, remove_non_speech(row['transcript']))


def test(lm):
    print('--test--')
    retest = True

    with open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        while retest:
            retest = False
            for i, row in enumerate(reader):
                file = '../data/corrupt/' + str(i).rjust(4, '0')

                if os.path.exists(file+'.pass') or os.path.exists(file+'.test'):
                    continue

                open(file + '.test', 'a').close()

                try:
                    data = lm.corrupt(i, remove_non_speech(row['transcript']))
                    if len(data):
                        assert(len(data[-1]) == 3)
                        assert(len(data[-1][0]) == 2)
                        assert(len(data[-1][1]) == 6)
                    else:
                        assert(data == [])
                    open(file + '.pass', 'a').close()

                except Exception as e:
                    print(i, "didn't work", e)
                    retest = True
                    if os.path.exists(file+'.bin'):
                        os.remove(file+'.bin')
                    lm.corrupt(i, remove_non_speech(row['transcript']))

                if os.path.exists(file + '.test'):
                    os.remove(file + '.test')


if __name__ == '__main__':
    # locker()
    print('gen_corrupt.py')
    lm = LanguageModel()
    gen(lm)
    gen(lm, nice=False)
    test(lm)


