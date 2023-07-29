import re
from random import shuffle


def save_dat(filename, version, data, columns, rows=None, append=False):
    file_location = '../data/test/' + filename + '.dat'

    if append:
        text = '\n' + ' '.join(data)
        try:
            with open(file_location, 'x') as file:
                text = ' '.join(columns) + text
                file.write(text)
        except FileExistsError:
            with open(file_location, 'a') as file:
                file.write(text)
        return

    with open(file_location, 'a'):
        pass

    with open(file_location, 'r') as file:
        firstline = file.read().split('\n')[0]

    if firstline == '' or int(firstline) < version:
        text = str(version) + '\n' + ' '.join(columns) + '\n'
        if rows is None:
            for i in range(len(data)):
                text = text + str(data[i]) + ' '
        else:
            for i in range(len(rows)):
                text = text + str(rows[i]).ljust(4, '0') + ' '
                for col in columns[1:]:
                    text = text + str(data[i][col]).ljust(16, '0') + ' '
                text = text + '\n'

        with open(file_location, 'w') as file:
            file.write(text)


def progress_bar(progresses, statuses, print_all=False):
    print('\n' * 50)
    head = [' ', '░', '▒', '▒']
    length = 50
    if print_all:
        for i in range(len(progresses)):
            progress = min(1, progresses[i])
            if progress == 1:
                continue
            partial = int(length * progress * 4) % 4
            bar = '█' * int(progress * length - 1) + head[partial]
            space = ' ' * (length - len(bar))
            print(str(i).rjust(3), '[%s%s]' % (bar, space), '{:.1%}'.format(progress).rjust(6), statuses[i])
    progress = sum(progresses) / len(progresses)
    partial = int(length * progress * 4) % 4
    bar = '█' * int(progress * length - 1) + head[partial]
    space = ' ' * (length - len(bar))
    print('all', '[%s%s]' % (bar, space), '{:.1%}'.format(progress).rjust(6))


def print_scores(scores, buckets):
    for i in range(min(scores), (max(scores) + 1), 1):
        print('{:.0%}'.format(i / buckets).rjust(4), str(scores[i]).rjust(4),
              '{0}'.format('█' * int(scores[i] / max(scores.values()) * 150)))
    print('\n')


def remove_non_speech(raw):
    speech = re.sub(r'\(.*?\)|<.*?>', '', raw)  # remove non-spoken cues
    return speech


def update_totals(totals, t, i, a):
    ti, ta = t == i, t == a
    tia = ti and ta
    totals = [a + b for a, b in zip(totals, [1, ti, ta, tia])]
    div_ti, div_ta, div_ia = totals[1] / totals[0], totals[2] / totals[0], totals[3] / totals[0]
    try:
        improvement = totals[2] / totals[1] - 1
    except ZeroDivisionError:
        improvement = 0

    print(str(ti).ljust(6) + '{:.2%}'.format(div_ti).rjust(6) + '\t' +
          str(ta).ljust(6) + '{:.2%}'.format(div_ta).rjust(6) + '\t' +
          str(tia).ljust(6) + '{:.2%}'.format(div_ia).rjust(6) + '\t',
          '|'+'{:.2%}'.format(improvement).rjust(8), '|', '\t', [t, i, a])

    return totals


def gen_naive_transcript(model, input, extra_words=4, preprocessed=False, verbose=True):
    if not len(input):
        return ''
    markers = [int(i/20*len(input)) for i in range(20)]

    if preprocessed:

        text = ' '.join(input[0][0])
        for i, data in enumerate(input):
            if verbose and i in markers:
                print('/', end='')
            text = text + ' ' + model.predict_word(data[0], data[1], verbose=False)
        return text

    text = ' '.join(list(input)[0][:-1])
    for i, ngram in enumerate([list(n) for n in input]):
        if verbose and i in markers:
            print('/', end='')
        prewords, words = ngram[:-1], ngram[-1:]
        words.extend(model.get_nrandom_words(extra_words, seed=str(ngram)))
        shuffle(words)
        word = model.predict_word(prewords, words, verbose=False)
        text = text + ' ' + word
    return text
