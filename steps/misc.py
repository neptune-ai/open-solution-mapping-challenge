from attrdict import AttrDict
import lightgbm as lgb
from sklearn.externals import joblib

from steps.base import BaseTransformer
from steps.utils import get_logger

logger = get_logger()


class LightGBM(BaseTransformer):
    def __init__(self, model_config, training_config):
        self.model_config = AttrDict(model_config)
        self.training_config = AttrDict(training_config)
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
        self.estimator = lgb.train(self.model_config,
                                   train, valid_sets=[train, valid], valid_names=['train', 'valid'],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self

    def transform(self, X, y=None, **kwargs):
        prediction = self.estimator.predict(X)
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.estimator, filepath)
