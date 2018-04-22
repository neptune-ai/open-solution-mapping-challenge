import os
from datetime import datetime, timedelta

from deepsense import neptune
from torch.optim.lr_scheduler import ExponentialLR

from steps.utils import get_logger
from .utils import Averager, save_model
from .validation import score_model

logger = get_logger()


class Callback:
    def __init__(self):
        self.epoch_id = None
        self.batch_id = None

        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.output_names = None
        self.validation_datagen = None
        self.lr_scheduler = None

    def set_params(self, transformer, validation_datagen):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0

    def on_train_end(self, *args, **kwargs):
        pass

    def on_epoch_begin(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        self.epoch_id += 1

    def training_break(self, *args, **kwargs):
        return False

    def on_batch_begin(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        self.batch_id += 1


class CallbackList:
    def __init__(self, callbacks=None):
        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = callbacks

    def __len__(self):
        return len(self.callbacks)

    def set_params(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.set_params(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(*args, **kwargs)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def training_break(self, *args, **kwargs):
        callback_out = [callback.training_break(*args, **kwargs) for callback in self.callbacks]
        return any(callback_out)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(*args, **kwargs)


class TrainingMonitor(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        self.epoch_loss_averagers = {}
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averagers = {}
        self.epoch_id = 0
        self.batch_id = 0

    def on_epoch_end(self, *args, **kwargs):
        for name, averager in self.epoch_loss_averagers.items():
            epoch_avg_loss = averager.value
            averager.reset()
            if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
                logger.info('epoch {0} {1}:     {2:.5f}'.format(self.epoch_id, name, epoch_avg_loss))
        self.epoch_id += 1

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]
            if name in self.epoch_loss_averagers.keys():
                self.epoch_loss_averagers[name].send(loss)
            else:
                self.epoch_loss_averagers[name] = Averager()
                self.epoch_loss_averagers[name].send(loss)

            if self.batch_every and ((self.batch_id % self.batch_every) == 0):
                logger.info('epoch {0} batch {1} {2}:     {3:.5f}'.format(self.epoch_id, self.batch_id, name, loss))
        self.batch_id += 1


class ValidationMonitor(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = score_model(self.model,
                                   self.loss_function,
                                   self.validation_datagen)
            self.model.train()
            for name, loss in val_loss.items():
                loss = loss.data.cpu().numpy()[0]
                logger.info('epoch {0} validation {1}:     {2:.5f}'.format(self.epoch_id, name, loss))
        self.epoch_id += 1


class EarlyStopping(Callback):
    def __init__(self, patience, minimize=True):
        super().__init__()
        self.patience = patience
        self.minimize = minimize
        self.best_score = None
        self.epoch_since_best = 0

    def training_break(self, *args, **kwargs):
        self.model.eval()
        val_loss = score_model(self.model, self.loss_function, self.validation_datagen)
        loss_sum = val_loss['sum']
        loss_sum = loss_sum.data.cpu().numpy()[0]

        self.model.train()

        if not self.best_score:
            self.best_score = loss_sum

        if (self.minimize and loss_sum < self.best_score) or (not self.minimize and loss_sum > self.best_score):
            self.best_score = loss_sum
            self.epoch_since_best = 0
        else:
            self.epoch_since_best += 1

        if self.epoch_since_best > self.patience:
            return True
        else:
            return False


class ExponentialLRScheduler(Callback):
    def __init__(self, gamma, epoch_every=1, batch_every=None):
        super().__init__()
        self.gamma = gamma
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def set_params(self, transformer, validation_datagen):
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.lr_scheduler = ExponentialLR(self.optimizer, self.gamma, last_epoch=-1)

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('initial lr: {0}'.format(self.optimizer.state_dict()['param_groups'][0]['initial_lr']))

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and (((self.epoch_id + 1) % self.epoch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} current lr: {1}'.format(self.epoch_id + 1,
                                                           self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.epoch_id += 1

    def on_batch_end(self, *args, **kwargs):
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} batch {1} current lr: {2}'.format(
                self.epoch_id + 1, self.batch_id + 1, self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.batch_id += 1


class ModelCheckpoint(Callback):
    def __init__(self, filepath, epoch_every=1, minimize=True):
        super().__init__()
        self.filepath = filepath
        self.minimize = minimize
        self.best_score = None

        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = score_model(self.model, self.loss_function, self.validation_datagen)
            loss_sum = val_loss['sum']
            loss_sum = loss_sum.data.cpu().numpy()[0]

            self.model.train()

            if not self.best_score:
                self.best_score = loss_sum

            if (self.minimize and loss_sum < self.best_score) or (not self.minimize and loss_sum > self.best_score):
                self.best_score = loss_sum
                save_model(self.model, self.filepath)
                logger.info('epoch {0} model saved to {1}'.format(self.epoch_id, self.filepath))

        self.epoch_id += 1


class NeptuneMonitor(Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.ctx = neptune.Context()
        self.epoch_loss_averager = Averager()

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averagers = {}
        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]

            if name in self.epoch_loss_averagers.keys():
                self.epoch_loss_averagers[name].send(loss)
            else:
                self.epoch_loss_averagers[name] = Averager()
                self.epoch_loss_averagers[name].send(loss)

            self.ctx.channel_send('{} batch {} loss'.format(self.model_name, name), x=self.batch_id, y=loss)

        self.batch_id += 1

    def on_epoch_end(self, *args, **kwargs):
        self._send_numeric_channels()
        self.epoch_id += 1

    def _send_numeric_channels(self, *args, **kwargs):
        for name, averager in self.epoch_loss_averagers.items():
            epoch_avg_loss = averager.value
            averager.reset()
            self.ctx.channel_send('{} epoch {} loss'.format(self.model_name, name), x=self.epoch_id, y=epoch_avg_loss)

        self.model.eval()
        val_loss = score_model(self.model,
                               self.loss_function,
                               self.validation_datagen)
        self.model.train()
        for name, loss in val_loss.items():
            loss = loss.data.cpu().numpy()[0]
            self.ctx.channel_send('{} epoch_val {} loss'.format(self.model_name, name), x=self.epoch_id, y=loss)


class ExperimentTiming(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every
        self.batch_start = None
        self.epoch_start = None
        self.current_sum = None
        self.current_mean = None

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('starting training...')

    def on_train_end(self, *args, **kwargs):
        logger.info('training finished')

    def on_epoch_begin(self, *args, **kwargs):
        if self.epoch_id > 0:
            epoch_time = datetime.now() - self.epoch_start
            if self.epoch_every:
                if (self.epoch_id % self.epoch_every) == 0:
                    logger.info('epoch {0} time {1}'.format(self.epoch_id - 1, str(epoch_time)[:-7]))
        self.epoch_start = datetime.now()
        self.current_sum = timedelta()
        self.current_mean = timedelta()
        logger.info('epoch {0} ...'.format(self.epoch_id))

    def on_batch_begin(self, *args, **kwargs):
        if self.batch_id > 0:
            current_delta = datetime.now() - self.batch_start
            self.current_sum += current_delta
            self.current_mean = self.current_sum / self.batch_id
        if self.batch_every:
            if self.batch_id > 0 and (((self.batch_id - 1) % self.batch_every) == 0):
                logger.info('epoch {0} average batch time: {1}'.format(self.epoch_id, str(self.current_mean)[:-5]))
        if self.batch_every:
            if self.batch_id == 0 or self.batch_id % self.batch_every == 0:
                logger.info('epoch {0} batch {1} ...'.format(self.epoch_id, self.batch_id))
        self.batch_start = datetime.now()


class ReduceLROnPlateau(Callback):  # thank you keras
    def __init__(self):
        super().__init__()
        pass
