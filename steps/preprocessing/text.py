import re
import string

import json
import numpy as np
import pandas as pd

from sklearn.externals import joblib
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from steps.base import BaseTransformer

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()
nltk.download('wordnet')
nltk.download('stopwords')
eng_stopwords = set(stopwords.words("english"))
with open('steps/resources/apostrophes.json', 'r') as f:
    APPO = json.load(f)


class WordListFilter(BaseTransformer):
    def __init__(self, word_list_filepath):
        self.word_set = self._read_data(word_list_filepath)

    def transform(self, X):
        X = self._transform(X)
        return {'X': X}

    def _transform(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X['text'] = X['text'].apply(self._filter_words)
        return X['text'].values

    def _filter_words(self, x):
        x = x.lower()
        x = ' '.join([w for w in x.split() if w in self.word_set])
        return x

    def _read_data(self, filepath):
        with open(filepath, 'r+') as f:
            data = f.read()
        return set(data.split('\n'))

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class TextCleaner(BaseTransformer):
    def __init__(self,
                 drop_punctuation,
                 drop_newline,
                 drop_multispaces,
                 all_lower_case,
                 fill_na_with,
                 deduplication_threshold,
                 anonymize,
                 apostrophes,
                 use_stopwords):
        self.drop_punctuation = drop_punctuation
        self.drop_newline = drop_newline
        self.drop_multispaces = drop_multispaces
        self.all_lower_case = all_lower_case
        self.fill_na_with = fill_na_with
        self.deduplication_threshold = deduplication_threshold
        self.anonymize = anonymize
        self.apostrophes = apostrophes
        self.use_stopwords = use_stopwords

    def transform(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X['text'] = X['text'].apply(self._transform)
        if self.fill_na_with:
            X['text'] = X['text'].fillna(self.fill_na_with).values
        return {'X': X['text'].values}

    def _transform(self, x):
        if self.all_lower_case:
            x = self._lower(x)
        if self.drop_punctuation:
            x = self._remove_punctuation(x)
        if self.drop_newline:
            x = self._remove_newline(x)
        if self.drop_multispaces:
            x = self._substitute_multiple_spaces(x)
        if self.deduplication_threshold is not None:
            x = self._deduplicate(x)
        if self.anonymize:
            x = self._anonymize(x)
        if self.apostrophes:
            x = self._apostrophes(x)
        if self.use_stopwords:
            x = self._use_stopwords(x)
        return x

    def _use_stopwords(self, x):
        words = tokenizer.tokenize(x)
        words = [w for w in words if not w in eng_stopwords]
        x = " ".join(words)
        return x

    def _apostrophes(self, x):
        words = tokenizer.tokenize(x)
        words = [APPO[word] if word in APPO else word for word in words]
        words = [lem.lemmatize(word, "v") for word in words]
        words = [w for w in words if not w in eng_stopwords]
        x = " ".join(words)
        return x

    def _anonymize(self, x):
        # remove leaky elements like ip,user
        x = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", x)
        # removing usernames
        x = re.sub("\[\[.*\]", " ", x)
        return x

    def _lower(self, x):
        return x.lower()

    def _remove_punctuation(self, x):
        return re.sub(r'[^\w\s]', ' ', x)

    def _remove_newline(self, x):
        x = x.replace('\n', ' ')
        x = x.replace('\n\n', ' ')
        return x

    def _substitute_multiple_spaces(self, x):
        return ' '.join(x.split())

    def _deduplicate(self, x):
        word_list = x.split()
        num_words = len(word_list)
        if num_words == 0:
            return x
        else:
            num_unique_words = len(set(word_list))
            unique_ratio = num_words / num_unique_words
            if unique_ratio > self.deduplication_threshold:
                x = ' '.join(x.split()[:num_unique_words])
            return x

    def load(self, filepath):
        params = joblib.load(filepath)
        self.drop_punctuation = params['drop_punctuation']
        self.all_lower_case = params['all_lower_case']
        self.fill_na_with = params['fill_na_with']
        return self

    def save(self, filepath):
        params = {'drop_punctuation': self.drop_punctuation,
                  'all_lower_case': self.all_lower_case,
                  'fill_na_with': self.fill_na_with,
                  }
        joblib.dump(params, filepath)


class TextCounter(BaseTransformer):
    def transform(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X = X['text'].apply(self._transform)
        X['caps_vs_length'] = self._caps_vs_length(X)
        X['num_symbols'] = X['text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
        X['num_words'] = X['text'].apply(lambda comment: len(comment.split()))
        X['num_unique_words'] = X['text'].apply(lambda comment: len(set(w for w in comment.split())))
        X['words_vs_unique'] = self._words_vs_unique(X)
        X['mean_word_len'] = X['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        X.drop('text', axis=1, inplace=True)
        X.fillna(0.0, inplace=True)
        return {'X': X}

    def _transform(self, x):
        features = {}
        features['text'] = x
        features['char_count'] = char_count(x)
        features['word_count'] = word_count(x)
        features['punctuation_count'] = punctuation_count(x)
        features['upper_case_count'] = upper_case_count(x)
        features['lower_case_count'] = lower_case_count(x)
        features['digit_count'] = digit_count(x)
        features['space_count'] = space_count(x)
        features['newline_count'] = newline_count(x)
        return pd.Series(features)

    def _caps_vs_length(self, X):
        try:
            return X.apply(lambda row: float(row['upper_case_count']) / float(row['char_count']), axis=1)
        except ZeroDivisionError:
            return 0

    def _words_vs_unique(self, X):
        try:
            return X['num_unique_words'] / X['num_words']
        except ZeroDivisionError:
            return 0

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


def char_count(x):
    return len(x)


def word_count(x):
    return len(x.split())


def newline_count(x):
    return x.count('\n')


def upper_case_count(x):
    return sum(c.isupper() for c in x)


def lower_case_count(x):
    return sum(c.islower() for c in x)


def digit_count(x):
    return sum(c.isdigit() for c in x)


def space_count(x):
    return sum(c.isspace() for c in x)


def punctuation_count(x):
    return occurence(x, string.punctuation)


def occurence(s1, s2):
    return sum([1 for x in s1 if x in s2])
