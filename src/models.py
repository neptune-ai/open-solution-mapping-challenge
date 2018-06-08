from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from .callbacks import NeptuneMonitorSegmentation, ValidationMonitorSegmentation
from .steps.pytorch.architectures.unet import UNet
from .steps.pytorch.callbacks import CallbackList, TrainingMonitor, ModelCheckpoint, \
    ExperimentTiming, ExponentialLRScheduler, EarlyStopping
from .steps.pytorch.models import Model
from .steps.pytorch.validation import multiclass_segmentation_loss, DiceLoss
from .utils import softmax
from .unet_models import AlbuNet, UNet11, UNet16, UNetResNet

PRETRAINED_NETWORKS = {'VGG11': {'model': UNet11,
                                 'model_config': {'num_classes': 2, 'pretrained': True},
                                 'init_weights': False},
                       'VGG16': {'model': UNet16,
                                 'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                                 'init_weights': False},
                       'AlbuNet': {'model': AlbuNet,
                                   'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                                   'init_weights': False},
                       'ResNet34': {'model': UNetResNet,
                                    'model_config': {'encoder_depth': 34, 'num_classes': 2,
                                                     'num_filters': 32, 'dropout_2d': 0.0,
                                                     'pretrained': True, 'is_deconv': True, },
                                    'init_weights': False},
                       'ResNet101': {'model': UNetResNet,
                                     'model_config': {'encoder_depth': 101, 'num_classes': 2,
                                                      'num_filters': 32, 'dropout_2d': 0.0,
                                                      'pretrained': True, 'is_deconv': True, },
                                     'init_weights': False},
                       'ResNet152': {'model': UNetResNet,
                                     'model_config': {'encoder_depth': 152, 'num_classes': 2,
                                                      'num_filters': 32, 'dropout_2d': 0.0,
                                                      'pretrained': True, 'is_deconv': True, },
                                     'init_weights': False}
                       }


class BasePyTorchUNet(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.set_model()
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = None
        self.callbacks = callbacks_unet(self.callbacks_config)

    def fit(self, datagen, validation_datagen=None, meta_valid=None):
        self._initialize_model_weights()

        self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.callbacks.set_params(self, validation_datagen=validation_datagen, meta_valid=meta_valid)
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

    def transform(self, datagen, validation_datagen=None, *args, **kwargs):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            outputs[name] = softmax(prediction, axis=1)
        return outputs

    def set_model(self):
        encoder = self.architecture_config['model_params']['encoder']
        if encoder == 'from_scratch':
            self.model = UNet(**self.architecture_config['model_params'])
        else:
            config = PRETRAINED_NETWORKS[encoder]
            self.model = config['model'](**config['model_config'])
            self._initialize_model_weights = lambda: None


class PyTorchUNet(BasePyTorchUNet):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.loss_function = [('multichannel_map', multiclass_segmentation_loss, 1.0)]


class PyTorchUNetStream(BasePyTorchUNet):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.loss_function = [('multichannel_map', multiclass_segmentation_loss, 1.0)]

    def transform(self, datagen, validation_datagen=None, *args, **kwargs):
        if len(self.output_names) == 1:
            output_generator = self._transform(datagen, validation_datagen)
            output = {'{}_prediction'.format(self.output_names[0]): output_generator}
            return output
        else:
            raise NotImplementedError

    def _transform(self, datagen, validation_datagen=None):
        self.model.eval()
        batch_gen, steps = datagen
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
            outputs_batch = outputs_batch.data.cpu().numpy()

            for output in outputs_batch:
                output = softmax(output, axis=0)
                yield output

            if batch_id == steps:
                break
        self.model.train()


class PyTorchUNetWeighted(BasePyTorchUNet):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        weighted_loss = partial(multiclass_weighted_cross_entropy,
                                **get_loss_variables(**architecture_config['weighted_cross_entropy']))
        loss = partial(mixed_dice_cross_entropy_loss, dice_weight=architecture_config['loss_weights']['dice_mask'],
                       cross_entropy_weight=architecture_config['loss_weights']['bce_mask'],
                       cross_entropy_loss=weighted_loss,
                       **architecture_config['dice'])
        self.loss_function = [('multichannel_map', loss, 1.0)]

    def transform(self, datagen, validation_datagen=None, *args, **kwargs):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            outputs[name] = softmax(prediction, axis=1)
        return outputs


class PyTorchUNetWeightedStream(BasePyTorchUNet):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        weighted_loss = partial(multiclass_weighted_cross_entropy,
                                **get_loss_variables(**architecture_config['weighted_cross_entropy']))
        loss = partial(mixed_dice_cross_entropy_loss, dice_weight=architecture_config['loss_weights']['dice_mask'],
                       cross_entropy_weight=architecture_config['loss_weights']['bce_mask'],
                       cross_entropy_loss=weighted_loss,
                       **architecture_config['dice'])
        self.loss_function = [('multichannel_map', loss, 1.0)]

    def transform(self, datagen, validation_datagen=None, *args, **kwargs):
        if len(self.output_names) == 1:
            output_generator = self._transform(datagen, validation_datagen)
            output = {'{}_prediction'.format(self.output_names[0]): output_generator}
            return output
        else:
            raise NotImplementedError

    def _transform(self, datagen, validation_datagen=None):
        self.model.eval()
        batch_gen, steps = datagen
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
            outputs_batch = outputs_batch.data.cpu().numpy()

            for output in outputs_batch:
                output = softmax(output, axis=0)
                yield output

            if batch_id == steps:
                break
        self.model.train()


def weight_regularization_unet(model, regularize, weight_decay_conv2d):
    if regularize:
        parameter_list = [{'params': model.parameters(), 'weight_decay': weight_decay_conv2d}]
    else:
        parameter_list = [model.parameters()]
    return parameter_list


def callbacks_unet(callbacks_config):
    experiment_timing = ExperimentTiming(**callbacks_config['experiment_timing'])
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['exp_lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitorSegmentation(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitorSegmentation(**callbacks_config['neptune_monitor'])
    early_stopping = EarlyStopping(**callbacks_config['early_stopping'])

    return CallbackList(
        callbacks=[experiment_timing, training_monitor, validation_monitor,
                   model_checkpoints, lr_scheduler, early_stopping, neptune_monitor,
                   ])


def multiclass_weighted_cross_entropy(output, target, w0, sigma, C):
    '''
    w1 is temporarily torch.ones - it should handle class imbalance for thw hole dataset
    '''
    distances = target[:, 1, :, :]
    sizes = target[:, 2, :, :]
    target = target[:, 0, :, :].long()

    w1 = Variable(torch.ones(distances.size()), requires_grad=False)  # TODO: fix it to handle class imbalance
    if torch.cuda.is_available():
        w1 = w1.cuda()
    size_weights = _get_size_weights(sizes, C)

    distance_weights = _get_distance_weights(distances, w1, w0, sigma)

    weights = distance_weights * size_weights
    loss_per_pixel = torch.nn.CrossEntropyLoss(reduce=False)(output, target)

    loss = torch.mean(loss_per_pixel * weights)
    return loss


def _get_distance_weights(d, w1, w0, sigma):
    weights = w1 + w0 * torch.exp(-(d ** 2) / (sigma ** 2))
    weights[d == 0] = 1
    return weights


def _get_size_weights(sizes, C):
    sizes_ = sizes.clone()
    sizes_[sizes == 0] = 1
    size_weights = C / sizes_
    size_weights[sizes_ == 1] = 1
    return size_weights


def get_loss_variables(w0, sigma, imsize):
    w0 = Variable(torch.Tensor([w0]), requires_grad=False)
    sigma = Variable(torch.Tensor([sigma]), requires_grad=False)
    C = Variable(torch.sqrt(torch.Tensor([imsize[0] * imsize[1]])) / 2, requires_grad=False)
    if torch.cuda.is_available():
        w0 = w0.cuda()
        sigma = sigma.cuda()
        C = C.cuda()
    return {'w0': w0, 'sigma': sigma, 'C': C}


def mixed_dice_cross_entropy_loss(output, target, dice_weight, cross_entropy_weight, cross_entropy_loss=None, smooth=0,
                                  dice_activation='softmax'):
    dice_target = target[:, 0, :, :].long()
    cross_entropy_target = target
    if cross_entropy_loss is None:
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        cross_entropy_target = dice_target
    return dice_weight * dice_loss(output, dice_target, smooth,
                                   dice_activation) + cross_entropy_weight * cross_entropy_loss(output,
                                                                                                cross_entropy_target)


def dice_loss(output, target, smooth=0, activation='softmax'):
    if activation == 'softmax':
        activation_nn = torch.nn.Softmax2d()
    elif activation == 'sigmoid':
        activation_nn = torch.nn.Sigmoid()
    else:
        raise NotImplementedError('only sigmoid and softmax are implemented')

    loss = 0
    dice = DiceLoss(smooth=smooth)
    output = activation_nn(output)
    for class_nr in range(1, int(target.max()) + 1):
        class_target = (target == class_nr)
        class_target.data = class_target.data.float()
        loss += dice(output[:, class_nr, :, :], class_target)
    return loss
