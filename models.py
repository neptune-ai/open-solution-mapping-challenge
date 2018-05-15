import torch
from torch.autograd import Variable
from torch import optim

from callbacks import NeptuneMonitorSegmentation
from steps.pytorch.architectures.unet import UNet
from steps.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    ExperimentTiming, ExponentialLRScheduler, EarlyStopping
from steps.pytorch.models import Model
from steps.pytorch.validation import multiclass_segmentation_loss
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
