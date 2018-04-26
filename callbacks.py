import numpy as np
import torch
from PIL import Image
from deepsense import neptune
from torch.autograd import Variable

from steps.pytorch.callbacks import NeptuneMonitor
from utils import softmax, categorize_image


class NeptuneMonitorSegmentation(NeptuneMonitor):
    def __init__(self, image_nr, image_resize, model_name, outputs_to_plot):
        super().__init__(model_name)
        self.image_nr = image_nr
        self.image_resize = image_resize
        self.outputs_to_plot = outputs_to_plot

    def on_epoch_end(self, *args, **kwargs):
        self._send_numeric_channels()
        self._send_image_channels()
        self.epoch_id += 1

    def _send_image_channels(self):
        self.model.eval()
        pred_masks = self.get_prediction_masks()
        self.model.train()

        for name, pred_mask in pred_masks.items():
            for i, image_duplet in enumerate(pred_mask):
                h, w = image_duplet.shape[1:]
                image_glued = np.zeros((h, 2 * w + 10))

                image_glued[:, :w] = image_duplet[0, :, :]
                image_glued[:, (w + 10):] = image_duplet[1, :, :]

                pill_image = Image.fromarray((image_glued * 255.).astype(np.uint8))
                h_, w_ = image_glued.shape
                pill_image = pill_image.resize((int(self.image_resize * w_), int(self.image_resize * h_)),
                                               Image.ANTIALIAS)

                self.ctx.channel_send('{} {}'.format(self.model_name, name), neptune.Image(
                    name='epoch{}_batch{}_idx{}'.format(self.epoch_id, self.batch_id, i),
                    description="true and prediction masks",
                    data=pill_image))

                if i == self.image_nr:
                    break

    def get_prediction_masks(self):
        prediction_masks = {}
        batch_gen, steps = self.validation_datagen
        for batch_id, data in enumerate(batch_gen):
            if len(data) != len(self.output_names) + 1:
                raise ValueError('incorrect targets provided')
            X = data[0]
            targets_tensors = data[1:]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)

            outputs_batch = self.model(X)
            if len(outputs_batch) == len(self.output_names):
                for name, output, target in zip(self.output_names, outputs_batch, targets_tensors):
                    if name not in self.outputs_to_plot:
                        continue
                    prediction = softmax(np.squeeze(output.data.cpu().numpy(), axis=1))
                    ground_truth = np.squeeze(target.cpu().numpy(), axis=1)
                    prediction_masks[name] = np.stack([prediction, ground_truth], axis=1)
            else:
                for name, target in zip(self.output_names, targets_tensors):
                    if name not in self.outputs_to_plot:
                        continue
                    prediction = categorize_image(softmax(outputs_batch.data.cpu().numpy()), channel_axis=1)
                    ground_truth = np.squeeze(target.cpu().numpy(), axis=1)
                    prediction_masks[name] = np.stack([prediction, ground_truth], axis=1)
            break
        return prediction_masks
