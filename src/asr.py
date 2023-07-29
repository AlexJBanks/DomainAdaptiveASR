import os.path
import pickle

from nltk import FreqDist, wordpunct_tokenize
from nltk.corpus import stopwords

import ngram
import word2vec
from util import remove_non_speech

_activation_functions = ['Ident', 'Offset', 'Step', 'Sigm', 'Gauss', 'SiLU', 'ReLU', 'DReLU']      # Do not change order


def multiplier(m, x):
    return x / m * (m + 1 - x)


def activation(x, param, func):
    if func == 'Ident':
        return abs(x)                                                   # |x|
    if func == 'Offset':
        return abs(x) - param                                           # |x|-param
    if func == 'Step':
        return abs(x) > param and 1 or 0                                # {0<|x|<param: 0, param<|x|<1: 1}
    if func == 'Sigm':
        return 1 / (1 + pow(2, 20 * (param - abs(x))))                  # 1/(1+2^{20(param-|x|)}
    if func == 'Gauss':
        return pow(2, -20 * ((abs(x) - param) ** 2))                    # 2^{-20(|x|-param)^2}
    if func == 'SiLU':
        return (abs(x) - param) / (1 + pow(2, 10 * (param - abs(x))))   # (|x|-param)/(1+2^{10(param-|x|)})
    if func == 'ReLU':
        return max(0, abs(x) - param)                                   # max(0,|x|-param)
    if func == 'DReLU':
        return abs(x) > param and abs(x) or 0                           # {0<|x|<param: 0, param<|x|<1: |x|}
    print('activation function', func, 'not recognised')
    exit(1)


class LanguageModel:
    def __init__(self):
        print('\tinit LanguageModel')
        ngram.load_corpora()
        self.ngram = ngram.generate_ngram_model()
        self.w2v = None
        self.stop_words = stopwords.words('english')
        print('\tLanguageModel ready')

    def extract_domain(self, raw, param=[0.5], funcs=['Step'], verbose=True):
        if self.w2v is None:
            self.w2v = word2vec.generate_word2vec_model()

        text = wordpunct_tokenize(remove_non_speech(raw))
        text = FreqDist(text)

        unusual_score = {}
        for w in [w for w in text if (not w.lower() in self.stop_words) and
                                     (len(w) > 1) and (w in self.w2v.key_to_index)]:
            unusual_score[w] = max(0, text.freq(w) - self.ngram.score(w))
        words = sorted(unusual_score, key=unusual_score.get, reverse=True)

        if funcs in [['Step'], ['ReLU'], ['DReLU']] and param == [1]:
            if len(words) == 0:
                return ''
            return words[0]

        max_freq = -1 if not len(words) else text[words[0]]

        sim_score = [{key: {} for key in funcs} for _ in range(len(param))]
        for v in words:
            # sim_score[v] = sum(activation(self.w2v.similarity(w, v), param, func) * unusual_score[w]
            # * multiplier(max_freq, text[w]) for w in words if v != w)
            modifier_term = unusual_score[v] * multiplier(max_freq, text[v])
            for w in [w for w in words if w != v]:
                w2v_sim = self.w2v.similarity(w, v)
                if verbose:
                    print(v, w, sim_score)

                for i in range(len(param)):
                    for func in funcs:
                        activ = activation(w2v_sim, param[i], func)
                        sim_score[i][func][w] = sim_score[i][func].get(w, 0) + activ * modifier_term

        domain = [{} for _ in range(len(param))]
        for i in range(len(param)):
            for func in funcs:
                domain[i][func] = max(sim_score[i][func], key=sim_score[i][func].get, default='')
                if verbose:
                    print(i, func, '\n\t', sim_score[i][func], '\n\t', 'domain:', domain[i][func])

        if len(param) == 1 and len(funcs) == 1:
            return domain[0][funcs[0]]
        return domain

    def predict_word(self, prewords, words, domain=None, verbose=True, probability=False):
        if self.w2v is None and domain is not None:
            self.w2v = word2vec.generate_word2vec_model()
        score = {}
        for w in words:
            score[w] = self.ngram.score(w, prewords) + 0.001 * self.ngram.score(w, prewords[1:])
            if domain is not None:
                if w not in self.w2v.key_to_index:
                    score[w] = score[w] * 0.001
                else:
                    score[w] = score[w] * self.w2v.similarity(w, domain)
            if probability:
                return score[w]
            if verbose:
                print(w, score[w])

        return max(score, key=score.get, default='')

    def evaluate(self, domain, topics):
        if self.w2v is None:
            self.w2v = word2vec.generate_word2vec_model()
        topics = [t for t in topics if t in self.w2v.key_to_index]
        if domain == '' or len(topics) == 0:
            return 1
        return max(self.w2v.similarity(domain, topic) for topic in topics)

    # pass through functions

    def generate_ngrams(self, transcript):
        return ngram.generate_ngrams(transcript)

    def get_nrandom_words(self, n, seed=''):
        return ngram.get_nrandom_words(self.ngram, n, seed)

    def corrupt(self, i, speech, verbose=True):
        file_name = str(i).rjust(4, '0')
        file_path = '../data/corrupt/' + file_name
        try:
            with open(file_path + '.bin', 'rb') as corrupt_bin:
                corrupt = pickle.load(corrupt_bin)
                if verbose:
                    print('found', file_name)

        except Exception as e:
            if verbose:
                print(file_name, 'Exception:', e, '| Generating', end=' ')
            open(file_path + '.lock', 'a').close()
            corrupt = ngram.corrupt(self.ngram, speech, verbose=verbose)
            if verbose:
                print(' saving')
            with open(file_path + '.bin', 'wb') as corrupt_bin:
                pickle.dump(corrupt, corrupt_bin)

        if os.path.exists(file_path + '.lock'):
            os.remove(file_path + '.lock')
        return corrupt
