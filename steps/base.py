import os
import shutil
import pprint

import numpy as np
from scipy import sparse
from sklearn.externals import joblib

from .utils import view_graph, plot_graph, get_logger, initialize_logger

initialize_logger()
logger = get_logger()


class Step:
    def __init__(self, name, transformer, input_steps=[], input_data=[], adapter=None,
                 cache_dirpath=None, cache_output=False, save_output=False, load_saved_output=False,
                 save_graph=False, force_fitting=False):
        self.name = name

        self.transformer = transformer

        self.input_steps = input_steps
        self.input_data = input_data
        self.adapter = adapter

        self.force_fitting = force_fitting
        self.cache_output = cache_output
        self.save_output = save_output
        self.load_saved_output = load_saved_output

        self.cache_dirpath = cache_dirpath
        self._prep_cache(cache_dirpath)

        if save_graph:
            graph_filepath = os.path.join(self.cache_dirpath, '{}_graph.json'.format(self.name))
            logger.info('Saving graph to {}'.format(graph_filepath))
            joblib.dump(self.graph_info, graph_filepath)

    def _copy_transformer(self, step, name, dirpath):
        self.transformer = self.transformer.transformer

        original_filepath = os.path.join(step.cache_dirpath, 'transformers', step.name)
        copy_filepath = os.path.join(dirpath, 'transformers', name)
        logger.info('copying transformer from {} to {}'.format(original_filepath, copy_filepath))
        shutil.copyfile(original_filepath, copy_filepath)

    def _prep_cache(self, cache_dirpath):
        for dirname in ['transformers', 'outputs', 'tmp']:
            os.makedirs(os.path.join(cache_dirpath, dirname), exist_ok=True)

        self.cache_dirpath_transformers = os.path.join(cache_dirpath, 'transformers')
        self.save_dirpath_outputs = os.path.join(cache_dirpath, 'outputs')
        self.save_dirpath_tmp = os.path.join(cache_dirpath, 'tmp')

        self.cache_filepath_step_transformer = os.path.join(self.cache_dirpath_transformers, self.name)
        self.save_filepath_step_output = os.path.join(self.save_dirpath_outputs, '{}'.format(self.name))
        self.save_filepath_step_tmp = os.path.join(self.save_dirpath_tmp, '{}'.format(self.name))

    def clean_cache(self):
        for name, step in self.all_steps.items():
            step._clean_cache()

    def _clean_cache(self):
        if os.path.exists(self.save_filepath_step_tmp):
            os.remove(self.save_filepath_step_tmp)

    @property
    def named_steps(self):
        return {step.name: step for step in self.input_steps}

    def get_step(self, name):
        return self.all_steps[name]

    @property
    def transformer_is_cached(self):
        if isinstance(self.transformer, Step):
            self._copy_transformer(self.transformer, self.name, self.cache_dirpath)
        return os.path.exists(self.cache_filepath_step_transformer)

    @property
    def output_is_cached(self):
        return os.path.exists(self.save_filepath_step_tmp)

    @property
    def output_is_saved(self):
        return os.path.exists(self.save_filepath_step_output)

    def fit_transform(self, data):
        if self.output_is_cached and not self.force_fitting:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.save_filepath_step_tmp)
        elif self.output_is_saved and self.load_saved_output and not self.force_fitting:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.save_filepath_step_output)
        else:
            step_inputs = {}
            if self.input_data is not None:
                for input_data_part in self.input_data:
                    step_inputs[input_data_part] = data[input_data_part]

            for input_step in self.input_steps:
                step_inputs[input_step.name] = input_step.fit_transform(data)

            if self.adapter:
                step_inputs = self.adapt(step_inputs)
            else:
                step_inputs = self.unpack(step_inputs)
            step_output_data = self._cached_fit_transform(step_inputs)
        return step_output_data

    def _cached_fit_transform(self, step_inputs):
        if self.transformer_is_cached and not self.force_fitting:
            logger.info('step {} loading transformer...'.format(self.name))
            self.transformer.load(self.cache_filepath_step_transformer)
            logger.info('step {} transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            logger.info('step {} fitting and transforming...'.format(self.name))
            step_output_data = self.transformer.fit_transform(**step_inputs)
            logger.info('step {} saving transformer...'.format(self.name))
            self.transformer.save(self.cache_filepath_step_transformer)

        if self.cache_output:
            logger.info('step {} caching outputs...'.format(self.name))
            self._save_output(step_output_data, self.save_filepath_step_tmp)
        if self.save_output:
            logger.info('step {} saving outputs...'.format(self.name))
            self._save_output(step_output_data, self.save_filepath_step_output)
        return step_output_data

    def _load_output(self, filepath):
        return joblib.load(filepath)

    def _save_output(self, output_data, filepath):
        joblib.dump(output_data, filepath)

    def transform(self, data):
        if self.output_is_cached:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.save_filepath_step_tmp)
        elif self.output_is_saved and self.load_saved_output:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.save_filepath_step_output)
        else:
            step_inputs = {}
            if self.input_data is not None:
                for input_data_part in self.input_data:
                    step_inputs[input_data_part] = data[input_data_part]

            for input_step in self.input_steps:
                step_inputs[input_step.name] = input_step.fit_transform(data)

            if self.adapter:
                step_inputs = self.adapt(step_inputs)
            else:
                step_inputs = self.unpack(step_inputs)
            step_output_data = self._cached_transform(step_inputs)
        return step_output_data

    def _cached_transform(self, step_inputs):
        if self.transformer_is_cached:
            logger.info('step {} loading transformer...'.format(self.name))
            self.transformer.load(self.cache_filepath_step_transformer)
            logger.info('step {} transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            raise ValueError('No transformer cached {}'.format(self.name))
        if self.cache_output:
            logger.info('step {} caching outputs...'.format(self.name))
            self._save_output(step_output_data, self.save_filepath_step_tmp)
        if self.save_output:
            logger.info('step {} saving outputs...'.format(self.name))
            self._save_output(step_output_data, self.save_filepath_step_output)
        return step_output_data

    def adapt(self, step_inputs):
        logger.info('step {} adapting inputs'.format(self.name))
        adapted_steps = {}
        for adapted_name, mapping in self.adapter.items():
            if isinstance(mapping, str):
                adapted_steps[adapted_name] = step_inputs[mapping]
            else:
                if len(mapping) == 2:
                    (step_mapping, func) = mapping
                elif len(mapping) == 1:
                    step_mapping = mapping
                    func = identity_inputs
                else:
                    raise ValueError('wrong mapping specified')

                raw_inputs = [step_inputs[step_name][step_var] for step_name, step_var in step_mapping]
                adapted_steps[adapted_name] = func(raw_inputs)
        return adapted_steps

    def unpack(self, step_inputs):
        logger.info('step {} unpacking inputs'.format(self.name))
        unpacked_steps = {}
        for step_name, step_dict in step_inputs.items():
            unpacked_steps = {**unpacked_steps, **step_dict}
        return unpacked_steps

    @property
    def all_steps(self):
        all_steps = {}
        all_steps = self._get_steps(all_steps)
        return all_steps

    def _get_steps(self, all_steps):
        for input_step in self.input_steps:
            all_steps = input_step._get_steps(all_steps)
        all_steps[self.name] = self
        return all_steps

    @property
    def graph_info(self):
        graph_info = {'edges': set(),
                      'nodes': set()}

        graph_info = self._get_graph_info(graph_info)

        return graph_info

    def _get_graph_info(self, graph_info):
        for input_step in self.input_steps:
            graph_info = input_step._get_graph_info(graph_info)
            graph_info['edges'].add((input_step.name, self.name))
        graph_info['nodes'].add(self.name)
        for input_data in self.input_data:
            graph_info['nodes'].add(input_data)
            graph_info['edges'].add((input_data, self.name))
        return graph_info

    def plot_graph(self, filepath):
        plot_graph(self.graph_info, filepath)

    def __str__(self):
        return pprint.pformat(self.graph_info)

    def _repr_html_(self):
        return view_graph(self.graph_info)


class BaseTransformer:
    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return NotImplementedError

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class MockTransformer(BaseTransformer):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)


class Dummy(BaseTransformer):
    def transform(self, **kwargs):
        return kwargs


def to_tuple_inputs(inputs):
    return tuple(inputs)


def identity_inputs(inputs):
    return inputs[0]


def sparse_hstack_inputs(inputs):
    return sparse.hstack(inputs)


def hstack_inputs(inputs):
    return np.hstack(inputs)


def vstack_inputs(inputs):
    return np.vstack(inputs)


def stack_inputs(inputs):
    stacked = np.stack(inputs, axis=0)
    return stacked


def sum_inputs(inputs):
    stacked = np.stack(inputs, axis=0)
    return np.sum(stacked, axis=0)


def average_inputs(inputs):
    stacked = np.stack(inputs, axis=0)
    return np.mean(stacked, axis=0)


def exp_transform(inputs):
    return np.exp(inputs[0])
