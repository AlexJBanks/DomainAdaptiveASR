import os
import re
import shutil
import threading
import time
import csv
import urllib.request
from itertools import islice
from urllib.error import HTTPError

import moviepy.editor

from util import progress_bar

regex_pattern = r'(?::\\"(https?:\/\/.+\.mp4)\\")'
local_path = '../data/TED/'
storage_path = '/media/alexb/maxone/data/TED/'

threads_num = 20
percentage = 1

progresses = [0.0] * threads_num
statuses = [' '] * threads_num
none_found = []

lock = threading.Lock()


def count_duration(reader):
    count = len([f for f in os.listdir(storage_path) if os.path.isfile(storage_path + f) and '.wav' in f])
    dur_done = sum(int(row['duration']) for row in islice(reader, count))
    duration = dur_done + sum(int(row['duration']) for row in reader)
    store_done = sum(
        os.stat(storage_path + f).st_size for f in os.listdir(storage_path) if
        os.path.isfile(storage_path + f) and '.wav' in f)
    storage = duration / dur_done * store_done

    print('count:' + str(count), str(int(count / 40.16)) + '%')
    # print(str(int(storage)) + "bytes")
    # storage /= 1024
    # print(str(int(storage)) + "KB")
    # storage /= 1024
    # print(str(int(storage)) + "MB")
    storage /= (1024 ** 3)
    print(str(int(storage)) + "GB")


class ScraperDaemon(threading.Thread):
    def __init__(self, reader, target: int, index, download=False, process=False, move=False):
        threading.Thread.__init__(self)
        self.r = reader
        self.t: int = target
        self.i = index
        self.d = download
        self.p = process
        self.m = move

    def run(self):
        scraper(self.r, self.t, self.d, self.p, self.m, self.i)


def scraper(reader, target: int, download, process, move, index=0):

    if move:
        _storage_path = storage_path
    else:
        _storage_path = local_path

    regex_matcher = re.compile(regex_pattern)

    t = 0
    backoff = 1

    # qualities = {'high': 0, 'medium': 0, 'low': 0}

    current_row = 0

    for row in reader:
        statuses[index] = ' '
        current_row = current_row + 1
        progresses[index] = current_row / target
        if current_row > target:
            break

        filename = re.sub(r'[^A-Za-z0-9]+', '_', row['title']) + '.wav'
        if not os.path.isfile(_storage_path + filename):
            raw_html = None
            while raw_html is None:
                try:
                    time.sleep(t)
                    statuses[index] = 'w'
                    with lock:
                        statuses[index] = 'l'
                        raw_html = urllib.request.urlopen(row['url']).read().decode('utf8')
                    backoff = max(1, backoff - 1)
                    t = max(0, t - 0.01)
                except HTTPError as e:
                    statuses[index] = 'b'
                    if e.code == 429:
                        t += 0.1
                        backoff *= 1.5
                        # print('\r', name, 'sleep ' + str(backoff) + 's', end='')
                        time.sleep(backoff)
                    else:
                        # print('\r', name, row['title'], e, end='')
                        raw_html = ''

            result = regex_matcher.findall(raw_html)
            # if not result:
            #    result = re.search(r'"en":{.*?"high":"(?P<high>.*?)".*?"low":"(?P<low>.*?)"}', raw_html)

            if not result:
                statuses[index] = 'n'
                none_found.append('0:' + row['url'])
                current_row = current_row - 1
            elif len(result) > 1:
                statuses[index] = 'n'
                none_found.append(str(len(result)) + ':' + row['url'])
                current_row = current_row - 1
            else:
                # for q in qualities.keys():
                #    if q in result.groupdict().keys() and result.group(q) != 'null':
                #        # print('\r', q + ": " + result.group(q), end='')
                #        qualities[q] = qualities[q] + 1
                download_url = result[0]
                # print('\r', name, download_url, end='')
                #        break
                # print('\r', str(qualities) + ' unknown:' + str(len(none_found)), end='')

                if download:
                    statuses[index] = 'd'
                    urllib.request.urlretrieve(download_url, str(index) + '.mp4')

                if process:
                    statuses[index] = 'p'
                    video = moviepy.editor.VideoFileClip(str(index) + '.mp4')
                    video.audio.write_audiofile(local_path + filename, logger=None)

                if move:
                    statuses[index] = 'm'
                    # print('\r', name, 'moving', end='')
                    shutil.move(local_path + filename, _storage_path + filename)
        # time.sleep(1.01)


if __name__ == '__main__':

    try:
        count = sum(1 for line in open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig'))
        quota = int(count * percentage)

        with open('../data/TED/ted_talks_en.csv', newline='', encoding='utf-8-sig') as csvfile:

            reader = csv.DictReader(csvfile)
            # count_duration(reader)

            daemons = []

            for i in range(threads_num):
                target = int(quota / (threads_num - i))
                quota = quota - target
                d = ScraperDaemon(reader, target, i, True, True, True)
                d.daemon = True
                d.start()
                daemons.append(d)
                time.sleep(1.97)
                progress_bar(progresses, statuses, True)

            running = True
            while running:
                time.sleep(2)
                running = any(daemon.is_alive() for daemon in daemons)
                progress_bar(progresses, statuses, True)

            for i in daemons:
                i.join()

        print(str(len(none_found)) + ' not found:', none_found)

    except FileNotFoundError:
        print('[ERROR] ted_talks_en.csv is missing. Please check README.md for more information.')
        exit(1)
