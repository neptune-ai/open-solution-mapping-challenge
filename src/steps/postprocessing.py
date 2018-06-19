import numpy as np
import pandas as pd
from sklearn.externals import joblib

from .base import BaseTransformer


class ClassPredictor(BaseTransformer):
    def transform(self, prediction_proba):
        predictions_class = np.argmax(prediction_proba, axis=1)
        return {'y_pred': predictions_class}

    def load(self, filepath):
        return ClassPredictor()

    def save(self, filepath):
        joblib.dump({}, filepath)


class PredictionAverage(BaseTransformer):
    def __init__(self, weights=None):
        self.weights = weights

    def transform(self, prediction_proba_list):
        if self.weights is not None:
            reshaped_weights = self._reshape_weights(prediction_proba_list.shape)
            prediction_proba_list *= reshaped_weights
            avg_pred = np.sum(prediction_proba_list, axis=0)
        else:
            avg_pred = np.mean(prediction_proba_list, axis=0)
        return {'prediction_probability': avg_pred}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.weights = params['weights']
        return self

    def save(self, filepath):
        joblib.dump({'weights': self.weights}, filepath)

    def _reshape_weights(self, prediction_shape):
        dim = len(prediction_shape)
        reshape_dim = (-1,) + tuple([1] * (dim - 1))
        reshaped_weights = np.array(self.weights).reshape(reshape_dim)
        return reshaped_weights


class PredictionAverageUnstack(BaseTransformer):
    def transform(self, prediction_probability, id_list):
        df = pd.DataFrame(prediction_probability)
        df['id'] = id_list
        avg_pred = df.groupby('id').mean().reset_index().drop(['id'], axis=1).values
        return {'prediction_probability': avg_pred}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class ProbabilityCalibration(BaseTransformer):
    def __init__(self, power):
        self.power = power

    def fit(self, prediction_proba):
        return self

    def transform(self, prediction_probability):
        prediction_probability = np.array(prediction_probability) ** self.power
        return {'prediction_probability': prediction_probability}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)
