import re

from deepsense import neptune
from keras import backend as K
from keras.callbacks import Callback


class NeptuneMonitor(Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.ctx = neptune.Context()
        self.batch_loss_channel_name = get_correct_channel_name(self.ctx,
                                                                '{} Batch Log-loss training'.format(self.model_name))
        self.epoch_loss_channel_name = get_correct_channel_name(self.ctx,
                                                                '{} Log-loss training'.format(self.model_name))
        self.epoch_val_loss_channel_name = get_correct_channel_name(self.ctx,
                                                                    '{} Log-loss validation'.format(self.model_name))

        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1
        self.ctx.channel_send(self.batch_loss_channel_name, self.batch_id, logs['loss'])

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1
        self.ctx.channel_send(self.epoch_loss_channel_name, self.epoch_id, logs['loss'])
        self.ctx.channel_send(self.epoch_val_loss_channel_name, self.epoch_id, logs['val_loss'])


class ReduceLR(Callback):
    def __init__(self, gamma):
        self.gamma = gamma

    def on_epoch_end(self, epoch, logs={}):
        if self.gamma is not None:
            K.set_value(self.model.optimizer.lr, self.gamma * K.get_value(self.model.optimizer.lr))


class UnfreezeLayers(Callback):
    def __init__(self, unfreeze_on_epoch, from_layer=0, to_layer=1):
        self.unfreeze_on_epoch = unfreeze_on_epoch
        self.from_layer = from_layer
        self.to_layer = to_layer

        self.epoch_id = 0
        self.batch_id = 0

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch_id == self.unfreeze_on_epoch:
            for i, layer in enumerate(self.model.layers):
                if i >= self.from_layer and i <= self.to_layer:
                    layer.trainable = True
        self.epoch_id += 1


def get_correct_channel_name(ctx, name):
    channels_with_name = [channel for channel in ctx._experiment._channels if name in channel.name]
    if len(channels_with_name) == 0:
        return name
    else:
        channel_ids = [re.split('[^\d]', channel.name)[-1] for channel in channels_with_name]
        channel_ids = sorted([int(idx) if idx != '' else 0 for idx in channel_ids])
        last_id = channel_ids[-1]
        corrected_name = '{} {}'.format(name, last_id + 1)
        return corrected_name
