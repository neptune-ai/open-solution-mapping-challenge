import os
import shutil
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init

from steps.base import BaseTransformer
from steps.utils import get_logger
from .utils import save_model

logger = get_logger()


class Model(BaseTransformer):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__()
        self.architecture_config = architecture_config
        self.training_config = training_config
        self.callbacks_config = callbacks_config

        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.callbacks = None

    @property
    def output_names(self):
        return [name for (name, func, weight) in self.loss_function]

    def _initialize_model_weights(self):
        logger.info('initializing model weights...')
        weights_init_config = self.architecture_config['weights_init']

        if weights_init_config['function'] == 'normal':
            weights_init_func = partial(init_weights_normal, **weights_init_config['params'])
        elif weights_init_config['function'] == 'xavier':
            weights_init_func = init_weights_xavier
        elif weights_init_config['function'] == 'he':
            weights_init_func = init_weights_he
        else:
            raise NotImplementedError

        self.model.apply(weights_init_func)

    def fit(self, datagen, validation_datagen=None):
        self._initialize_model_weights()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model

        self.callbacks.set_params(self, validation_datagen=validation_datagen)
        self.callbacks.on_train_begin()

        batch_gen, steps = datagen
        for epoch_id in range(self.training_config['epochs']):
            self.callbacks.on_epoch_begin()
            for batch_id, data in enumerate(batch_gen):
                self.callbacks.on_batch_begin()
                metrics = self._fit_loop(data)
                self.callbacks.on_batch_end(metrics=metrics)
                if batch_id == steps:
                    break
            self.callbacks.on_epoch_end()
            if self.callbacks.training_break():
                break
        self.callbacks.on_train_end()
        return self

    def _fit_loop(self, data):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = Variable(X).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor).cuda())
        else:
            X = Variable(X)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor))

        self.optimizer.zero_grad()
        outputs_batch = self.model(X)
        partial_batch_losses = {}

        assert len(targets_tensors) == len(outputs_batch) == len(self.loss_function),\
            '''Number of targets, model outputs and elements of loss function must equal.
            You have n_targets={0}, n_model_outputs={1}, n_loss_function_elements={2}.
            The order of elements must also be preserved.'''.format(len(targets_tensors),
                                                                    len(outputs_batch),
                                                                    len(self.loss_function))

        if len(self.output_names) == 1:
            for (name, loss_function, weight), target in zip(self.loss_function, targets_var):
                batch_loss = loss_function(outputs_batch, target) * weight
        else:
            for (name, loss_function, weight), output, target in zip(self.loss_function, outputs_batch, targets_var):
                partial_batch_losses[name] = loss_function(output, target) * weight
            batch_loss = sum(partial_batch_losses.values())
        partial_batch_losses['sum'] = batch_loss
        batch_loss.backward()
        self.optimizer.step()

        return partial_batch_losses

    def _transform(self, datagen, validation_datagen=None):
        self.model.eval()
        batch_gen, steps = datagen
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            if isinstance(data, list):
                X = data[0]
            else:
                X = data

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)

            outputs_batch = self.model(X)
            if len(self.output_names) == 1:
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                for name, output in zip(self.output_names, outputs_batch):
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
            if batch_id == steps:
                break
        self.model.train()
        outputs = {'{}_prediction'.format(name): np.vstack(outputs_) for name, outputs_ in outputs.items()}
        return outputs

    def transform(self, datagen, validation_datagen=None):
        predictions = self._transform(datagen, validation_datagen)
        return NotImplementedError

    def load(self, filepath):
        self.model.eval()

        if torch.cuda.is_available():
            self.model.cpu()
            self.model.load_state_dict(torch.load(filepath))
            self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        return self

    def save(self, filepath):
        checkpoint_callback = self.callbacks_config.get('model_checkpoint')
        if checkpoint_callback:
            checkpoint_filepath = checkpoint_callback['filepath']
            if os.path.exists(checkpoint_filepath):
                shutil.copyfile(checkpoint_filepath, filepath)
            else:
                save_model(self.model, filepath)
        else:
            save_model(self.model, filepath)


class PyTorchBasic(nn.Module):
    def _flatten_features(self, in_size, features):
        f = features(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        features = self.features(x)
        flat_features = features.view(-1, self.flat_features)
        out = self.classifier(flat_features)
        return out

    def forward_target(self, x):
        return self.forward(x)


def init_weights_normal(model, mean, std_conv2d, std_linear):
    if type(model) == nn.Conv2d:
        model.weight.data.normal_(mean=mean, std=std_conv2d)
    if type(model) == nn.Linear:
        model.weight.data.normal_(mean=mean, std=std_linear)


def init_weights_xavier(model):
    if isinstance(model, nn.Conv2d):
        init.xavier_normal(model.weight)
        init.constant(model.bias, 0)
        
def init_weights_he(model):
    if isinstance(model, nn.Conv2d):
        init.kaiming_normal(model.weight)
        init.constant(model.bias, 0)
        
