import sklearn.decomposition as decomp
import sklearn.preprocessing as sk_prep
from sklearn.externals import joblib
from sklearn.feature_extraction import text

from ...steps.base import BaseTransformer


class XYSplit(BaseTransformer):
    def __init__(self, x_columns, y_columns):
        self.x_columns = x_columns
        self.y_columns = y_columns

    def transform(self, meta, train_mode):
        X = meta[self.x_columns].values
        if train_mode:
            y = meta[self.y_columns].values
        else:
            y = None

        return {'X': X,
                'y': y}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.columns_to_get = params['x_columns']
        self.target_columns = params['y_columns']
        return self

    def save(self, filepath):
        params = {'x_columns': self.x_columns,
                  'y_columns': self.y_columns
                  }
        joblib.dump(params, filepath)


class TfidfVectorizer(BaseTransformer):
    def __init__(self, **kwargs):
        self.vectorizer = text.TfidfVectorizer(**kwargs)

    def fit(self, text):
        self.vectorizer.fit(text)
        return self

    def transform(self, text):
        return {'features': self.vectorizer.transform(text)}

    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.vectorizer, filepath)


class TruncatedSVD(BaseTransformer):
    def __init__(self, **kwargs):
        self.truncated_svd = decomp.TruncatedSVD(**kwargs)

    def fit(self, features):
        self.truncated_svd.fit(features)
        return self

    def transform(self, features):
        return {'features': self.truncated_svd.transform(features)}

    def load(self, filepath):
        self.truncated_svd = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.truncated_svd, filepath)


class Normalizer(BaseTransformer):
    def __init__(self):
        self.normalizer = sk_prep.Normalizer()

    def fit(self, X):
        self.normalizer.fit(X)
        return self

    def transform(self, X):
        X = self.normalizer.transform(X)
        return {'X': X}

    def load(self, filepath):
        self.normalizer = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.normalizer, filepath)


class MinMaxScaler(BaseTransformer):
    def __init__(self):
        self.minmax_scaler = sk_prep.MinMaxScaler()

    def fit(self, X):
        self.minmax_scaler.fit(X)
        return self

    def transform(self, X):
        X = self.minmax_scaler.transform(X)
        return {'X': X}

    def load(self, filepath):
        self.minmax_scaler = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.minmax_scaler, filepath)


class MinMaxScalerMultilabel(BaseTransformer):
    def __init__(self):
        self.minmax_scalers = []

    def fit(self, X):
        for i in range(X.shape[1]):
            minmax_scaler = sk_prep.MinMaxScaler()
            minmax_scaler.fit(X[:, i, :])
            self.minmax_scalers.append(minmax_scaler)
        return self

    def transform(self, X):
        for i, minmax_scaler in enumerate(self.minmax_scalers):
            X[:, i, :] = minmax_scaler.transform(X[:, i, :])
        return {'X': X}

    def load(self, filepath):
        self.minmax_scalers = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.minmax_scalers, filepath)
