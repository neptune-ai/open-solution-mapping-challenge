from attrdict import AttrDict
import numpy as np
import sklearn.linear_model as lr
from sklearn import svm
from sklearn import ensemble
from sklearn.externals import joblib
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from steps.base import BaseTransformer
from steps.utils import get_logger

logger = get_logger()


class SklearnClassifier(BaseTransformer):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        prediction = self.estimator.predict_proba(X)
        return {'prediction': prediction}


class SklearnRegressor(BaseTransformer):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        prediction = self.estimator.predict(X)
        return {'prediction': prediction}


class SklearnTransformer(BaseTransformer):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        transformed = self.estimator.transform(X)
        return {'transformed': transformed}


class SklearnPipeline(BaseTransformer):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        transformed = self.estimator.transform(X)
        return {'transformed': transformed}


class LightGBM(BaseTransformer):
    def __init__(self, model_params, training_params):
        self.model_params = model_params
        self.training_params = AttrDict(training_params)
        self.evaluation_function = None

    def fit(self, X, y, X_valid, y_valid, feature_names, categorical_features, **kwargs):
        train = lgb.Dataset(X, label=y,
                            feature_name=feature_names,
                            categorical_feature=categorical_features
                            )
        valid = lgb.Dataset(X_valid, label=y_valid,
                            feature_name=feature_names,
                            categorical_feature=categorical_features
                            )

        evaluation_results = {}
        self.estimator = lgb.train(self.model_params,
                                   train,
                                   valid_sets=[train, valid],
                                   valid_names=['train', 'valid'],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_params.number_boosting_rounds,
                                   early_stopping_rounds=self.training_params.early_stopping_rounds,
                                   verbose_eval=10,
                                   feval=self.evaluation_function)
        return self

    def transform(self, X, y=None, **kwargs):
        prediction = self.estimator.predict(X)
        return {'prediction': prediction}


class MultilabelEstimator(BaseTransformer):
    def __init__(self, label_nr, **kwargs):
        self.label_nr = label_nr
        self.estimators = self._get_estimators(**kwargs)

    @property
    def estimator(self):
        return NotImplementedError

    def _get_estimators(self, **kwargs):
        estimators = []
        for i in range(self.label_nr):
            estimators.append((i, self.estimator(**kwargs)))
        return estimators

    def fit(self, X, y, **kwargs):
        for i, estimator in self.estimators:
            logger.info('fitting estimator {}'.format(i))
            estimator.fit(X, y[:, i])
        return self

    def transform(self, X, y=None, **kwargs):
        predictions = []
        for i, estimator in self.estimators:
            prediction = estimator.predict_proba(X)
            predictions.append(prediction)
        predictions = np.stack(predictions, axis=0)
        predictions = predictions[:, :, 1].transpose()
        return {'prediction_probability': predictions}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.label_nr = params['label_nr']
        self.estimators = params['estimators']
        return self

    def save(self, filepath):
        params = {'label_nr': self.label_nr,
                  'estimators': self.estimators}
        joblib.dump(params, filepath)


class LogisticRegressionMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return lr.LogisticRegression


class SVCMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return svm.SVC


class LinearSVC_proba(svm.LinearSVC):
    def __platt_func(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions).reshape(-1, 1)
        prob_positive = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        prob_negative = 1.0 - prob_positive
        probs = np.hstack([prob_negative, prob_positive])
        print(prob_positive)
        return probs


class LinearSVCMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return LinearSVC_proba


class RandomForestMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return ensemble.RandomForestClassifier


class CatboostClassifierMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return CatBoostClassifier


class XGBoostClassifierMultilabel(MultilabelEstimator):
    @property
    def estimator(self):
        return XGBClassifier


def make_transformer(estimator, mode='classifier'):
    if mode == 'classifier':
        transformer = SklearnClassifier(estimator)
    elif mode == 'regressor':
        transformer = SklearnRegressor(estimator)
    elif mode == 'transformer':
        transformer = SklearnTransformer(estimator)
    elif mode == 'pipeline':
        transformer = SklearnPipeline(estimator)
    else:
        raise NotImplementedError("""Only classifier, regressor and transformer modes are available""")

    return transformer
