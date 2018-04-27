import numpy as np
from gensim.models import KeyedVectors
from sklearn.externals import joblib

from ..base import BaseTransformer


class EmbeddingsMatrix(BaseTransformer):
    def __init__(self, pretrained_filepath, max_features, embedding_size):
        self.pretrained_filepath = pretrained_filepath
        self.max_features = max_features
        self.embedding_size = embedding_size

    def fit(self, tokenizer):
        self.embedding_matrix = self._get_embedding_matrix(tokenizer)
        return self

    def transform(self, tokenizer):
        return {'embeddings_matrix': self.embedding_matrix}

    def _get_embedding_matrix(self, tokenizer):
        return NotImplementedError

    def save(self, filepath):
        joblib.dump(self.embedding_matrix, filepath)

    def load(self, filepath):
        self.embedding_matrix = joblib.load(filepath)
        return self


class GloveEmbeddingsMatrix(EmbeddingsMatrix):
    def _get_embedding_matrix(self, tokenizer):
        return load_glove_embeddings(self.pretrained_filepath,
                                     tokenizer,
                                     self.max_features,
                                     self.embedding_size)


class Word2VecEmbeddingsMatrix(EmbeddingsMatrix):
    def _get_embedding_matrix(self, tokenizer):
        return load_word2vec_embeddings(self.pretrained_filepath,
                                        tokenizer,
                                        self.max_features,
                                        self.embedding_size)


class FastTextEmbeddingsMatrix(EmbeddingsMatrix):
    def _get_embedding_matrix(self, tokenizer):
        return load_fasttext_embeddings(self.pretrained_filepath,
                                        tokenizer,
                                        self.max_features,
                                        self.embedding_size)


def load_glove_embeddings(filepath, tokenizer, max_features, embedding_size):
    embeddings_index = dict()
    with open(filepath) as f:
        for line in f:
            # Note: use split(' ') instead of split() if you get an error.
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def load_word2vec_embeddings(filepath, tokenizer, max_features, embedding_size):
    model = KeyedVectors.load_word2vec_format(filepath, binary=True)

    emb_mean, emb_std = model.wv.syn0.mean(), model.wv.syn0.std()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        try:
            embedding_vector = model[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue
    return embedding_matrix


def load_fasttext_embeddings(filepath, tokenizer, max_features, embedding_size):
    embeddings_index = dict()
    with open(filepath) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                continue
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if coefs.shape[0] != embedding_size:
                continue
            embeddings_index[word] = coefs

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
