# used reference to https://radimrehurek.com/gensim/models/word2vec

import os.path

import gensim.downloader as gen_down
from gensim.models import KeyedVectors

data_path = '/home/alexb/git/fyp/data/word2vec/'


def generate_word2vec_model():
    model_name = 'word2vec-google-news-300'
    vector_path = data_path + model_name + '.vec'

    if os.path.exists(vector_path):
        print('4/4\tLoading vectors from', vector_path)
        model = KeyedVectors.load(vector_path)
    else:
        print('4/5\tDownloading and vectorising', model_name)
        print('5/5\tThis might take a while, but will speed up subsequent executions.')
        model = gen_down.load(model_name)
        model.save(vector_path)

    return model


def test(model):
    print('king vector:', model['king'])
    print('king vs queen similarity:', model.similarity('king', 'queen'))


if __name__ == '__main__':
    print('word2vec.py')
    model = generate_word2vec_model()
    test(model)
