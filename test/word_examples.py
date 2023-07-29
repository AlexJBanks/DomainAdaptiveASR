from asr import LanguageModel


def test(model):
    print('the, no domain:',
          model.predict_word(['the'], ['cat', 'dog', 'house', 'although', 'the']), '\n')
    print('the black, no domain:',
          model.predict_word(['the', 'black'], ['cat', 'dog', 'house', 'although', 'the']), '\n')
    print('the, property:',
          model.predict_word(['the'], ['cat', 'dog', 'house', 'although', 'the'], 'property'), '\n')
    print('the black, property:',
          model.predict_word(['the', 'black'], ['cat', 'dog', 'house', 'although', 'the'], 'property'), '\n')


if __name__ == '__main__':
    print('test/word_examples.py')
    lm = LanguageModel()
    test(lm)
