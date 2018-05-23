from functools import partial

import torch
from torch.autograd import Variable
from torch import optim

from callbacks import NeptuneMonitorSegmentation
from steps.pytorch.architectures.unet import UNet
from steps.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    ExperimentTiming, ExponentialLRScheduler, EarlyStopping
from steps.pytorch.models import Model
from steps.pytorch.validation import multiclass_segmentation_loss, DiceLoss
from utils import softmax
from unet_models import UNet11, UNet16, AlbuNet


class PyTorchUNet(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.set_model()
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = [('multichannel_map', multiclass_segmentation_loss, 1.0)]
        self.callbacks = callbacks_unet(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            outputs[name] = softmax(prediction, axis=1)
        return outputs

    def set_model(self):
        configs = {'VGG11': {'model': UNet11,
                             'model_config': {'num_classes': 2, 'pretrained': True},
                             'init_weights': False},
                   'VGG16': {'model': UNet16,
                             'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                             'init_weights': False},
                   'ResNet': {'model': AlbuNet,
                              'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                              'init_weights': False},
                   'standard': {'model': UNet,
                                'model_config': self.architecture_config['model_params'],
                                'init_weights': True}
                   }
        encoder = self.architecture_config['model_params']['encoder']
        config = configs[encoder]

        self.model = config['model'](**config['model_config'])
        if not config['init_weights']:
            self._initialize_model_weights = lambda: None


class PyTorchUNetStream(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.set_model()
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = [('multichannel_map', multiclass_segmentation_loss, 1.0)]
        self.callbacks = callbacks_unet(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        if len(self.output_names) == 1:
            output_generator = self._transform(datagen, validation_datagen)
            output = {'{}_prediction'.format(self.output_names[0]): output_generator}
            return output
        else:
            raise NotImplementedError

    def set_model(self):
        configs = {'VGG11': {'model': UNet11,
                             'model_config': {'num_classes': 2, 'pretrained': True},
                             'init_weights': False},
                   'VGG16': {'model': UNet16,
                             'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                             'init_weights': False},
                   'ResNet': {'model': AlbuNet,
                              'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                              'init_weights': False},
                   'standard': {'model': UNet,
                                'model_config': self.architecture_config['model_params'],
                                'init_weights': True}
                   }
        encoder = self.architecture_config['model_params']['encoder']
        config = configs[encoder]

        self.model = config['model'](**config['model_config'])
        if not config['init_weights']:
            self._initialize_model_weights = lambda: None

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


class PyTorchUNetWeighted(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.set_model()
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        weighted_loss = partial(multiclass_weighted_cross_entropy,
                                **get_loss_params(**training_config["loss_function"]))
        loss = partial(mixed_dice_cross_entropy_loss, dice_weight=architecture_config['loss_weights']['dice_mask'],
                       ce_weight=architecture_config['loss_weights']['bce_mask'], ce_loss=weighted_loss)
        self.loss_function = [('multichannel_map', loss, 1.0)]
        self.callbacks = callbacks_unet(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            outputs[name] = softmax(prediction, axis=1)
        return outputs

    def set_model(self):
        configs = {'VGG11': {'model': UNet11,
                             'model_config': {'num_classes': 2, 'pretrained': True},
                             'init_weights': False},
                   'VGG16': {'model': UNet16,
                             'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                             'init_weights': False},
                   'ResNet': {'model': AlbuNet,
                              'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                              'init_weights': False},
                   'standard': {'model': UNet,
                                'model_config': self.architecture_config['model_params'],
                                'init_weights': True}
                   }
        encoder = self.architecture_config['model_params']['encoder']
        config = configs[encoder]

        self.model = config['model'](**config['model_config'])
        if not config['init_weights']:
            self._initialize_model_weights = lambda: None


class PyTorchUNetWeightedStream(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.set_model()
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        loss = partial(multiclass_weighted_cross_entropy, **get_loss_params(**training_config["loss_function"]))
        self.loss_function = [('multichannel_map', loss, 1.0)]
        self.callbacks = callbacks_unet(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            outputs[name] = softmax(prediction, axis=1)
        return outputs

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

    def set_model(self):
        configs = {'VGG11': {'model': UNet11,
                             'model_config': {'num_classes': 2, 'pretrained': True},
                             'init_weights': False},
                   'VGG16': {'model': UNet16,
                             'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                             'init_weights': False},
                   'ResNet': {'model': AlbuNet,
                              'model_config': {'num_classes': 2, 'pretrained': True, 'is_deconv': True},
                              'init_weights': False},
                   'standard': {'model': UNet,
                                'model_config': self.architecture_config['model_params'],
                                'init_weights': True}
                   }
        encoder = self.architecture_config['model_params']['encoder']
        config = configs[encoder]

        self.model = config['model'](**config['model_config'])
        if not config['init_weights']:
            self._initialize_model_weights = lambda: None


def weight_regularization(model, regularize, weight_decay_conv2d, weight_decay_linear):
    if regularize:
        parameter_list = [{'params': model.features.parameters(), 'weight_decay': weight_decay_conv2d},
                          {'params': model.classifier.parameters(), 'weight_decay': weight_decay_linear},
                          ]
    else:
        parameter_list = [model.parameters()]
    return parameter_list


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
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
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
    size_weights = C / sizes
    size_weights[sizes == 1] = 1
    return size_weights


def get_loss_params(w0, sigma, imsize):
    w0 = Variable(torch.Tensor([w0]), requires_grad=False)
    sigma = Variable(torch.Tensor([sigma]), requires_grad=False)
    C = Variable(torch.sqrt(torch.Tensor([imsize[0] * imsize[1]])) / 2, requires_grad=False)
    if torch.cuda.is_available():
        w0 = w0.cuda()
        sigma = sigma.cuda()
        C = C.cuda()
    return {'w0': w0, 'sigma': sigma, 'C': C}


def mixed_dice_cross_entropy_loss(output, target, dice_weight, ce_weight, ce_loss=None):
    dice_target = target[:, 0, :, :].long()
    ce_target = target
    if ce_loss is None:
        ce_loss = torch.nn.CrossEntropyLoss()
        ce_target = dice_target
    return dice_weight * dice_loss(output, dice_target) + ce_weight * ce_loss(output, ce_target)


def dice_loss(output, target):
    dice_numerator = 0
    dice_denominator = 0
    loss = 0
    dice = DiceLoss()
    for class_nr in range(1, int(target.max()) + 1):
        class_target = (target == class_nr)
        class_target.data = class_target.data.float()
        #dice_numerator += (class_target * output[:, class_nr, :, :]).sum()
        #dice_denominator += (class_target + output[:, class_nr, :, :]).sum()
        loss += dice(output[:, class_nr, :, :], class_target)
    #return 1 - 2 * dice_numerator / dice_denominator
    return loss
