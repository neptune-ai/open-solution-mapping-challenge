from keras.preprocessing import text, sequence
from sklearn.externals import joblib

from ..base import BaseTransformer


class Tokenizer(BaseTransformer):
    def __init__(self, char_level, maxlen, num_words):
        self.char_level = char_level
        self.maxlen = maxlen
        self.num_words = num_words

        self.tokenizer = text.Tokenizer(char_level=self.char_level, num_words=self.num_words)

    def fit(self, X, X_valid=None, train_mode=True):
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X, X_valid=None, train_mode=True):
        X_tokenized = self._transform(X)

        if X_valid is not None:
            X_valid_tokenized = self._transform(X_valid)
        else:
            X_valid_tokenized = None
        return {'X': X_tokenized,
                'X_valid': X_valid_tokenized,
                'tokenizer': self.tokenizer}

    def _transform(self, X):
        list_tokenized = self.tokenizer.texts_to_sequences(list(X))
        X_tokenized = sequence.pad_sequences(list_tokenized, maxlen=self.maxlen)
        return X_tokenized

    def load(self, filepath):
        object_pickle = joblib.load(filepath)
        self.char_level = object_pickle['char_level']
        self.maxlen = object_pickle['maxlen']
        self.num_words = object_pickle['num_words']
        self.tokenizer = object_pickle['tokenizer']
        return self

    def save(self, filepath):
        object_pickle = {'char_level': self.char_level,
                         'maxlen': self.maxlen,
                         'num_words': self.num_words,
                         'tokenizer': self.tokenizer}
        joblib.dump(object_pickle, filepath)


class TextAugmenter(BaseTransformer):
    pass
    """
    Augmentations by Thesaurus synonim substitution or typos
    """
