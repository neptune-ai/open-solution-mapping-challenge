import os
import numpy as np
import torch
import json
from PIL import Image
from deepsense import neptune
from torch.autograd import Variable
from tempfile import TemporaryDirectory

from steps.utils import get_logger
from steps.pytorch.callbacks import NeptuneMonitor, ValidationMonitor
from utils import softmax, categorize_image, coco_evaluation, create_annotations
from pipeline_config import CATEGORY_IDS, Y_COLUMNS_SCORING

logger = get_logger()


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

            if (targets_tensors[0].size()[1] > 1):
                targets_tensors = [target_tensor[:, :1] for target_tensor in targets_tensors]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)

            outputs_batch = self.model(X)
            if len(outputs_batch) == len(self.output_names):
                for name, output, target in zip(self.output_names, outputs_batch, targets_tensors):
                    if name in self.outputs_to_plot:
                        prediction = categorize_image(softmax(output.data.cpu().numpy()), channel_axis=1)
                        ground_truth = np.squeeze(target.cpu().numpy(), axis=1)
                        n_channels = output.data.cpu().numpy().shape[1]
                        for channel_nr in range(n_channels):
                            category_id = CATEGORY_IDS[channel_nr]
                            if category_id != None:
                                channel_ground_truth = np.where(ground_truth == channel_nr, 1, 0)
                                mask_key = '{}_{}'.format(name, category_id)
                                prediction_masks[mask_key] = np.stack([prediction, channel_ground_truth], axis=1)
            else:
                for name, target in zip(self.output_names, targets_tensors):
                    if name in self.outputs_to_plot:
                        prediction = categorize_image(softmax(outputs_batch.data.cpu().numpy()), channel_axis=1)
                        ground_truth = np.squeeze(target.cpu().numpy(), axis=1)
                        n_channels = outputs_batch.data.cpu().numpy().shape[1]
                        for channel_nr in range(n_channels):
                            category_id = CATEGORY_IDS[channel_nr]
                            if category_id != None:
                                channel_ground_truth = np.where(ground_truth == channel_nr, 1, 0)
                                mask_key = '{}_{}'.format(name, category_id)
                                prediction_masks[mask_key] = np.stack([prediction, channel_ground_truth], axis=1)
            break
        return prediction_masks


class ValidationMonitorSegmentation(ValidationMonitor):
    def __init__(self, data_dir, validate_with_map=False, small_annotations_size=14, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.validate_with_map = validate_with_map
        self.small_annotations_size = small_annotations_size
        self.validation_loss = None
        self.validation_pipeline = None
        self.meta_valid = None

    def set_params(self, transformer, validation_datagen, *args, meta_valid=None, **kwargs):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen
        self.validation_loss = transformer.validation_loss
        self.validation_pipeline = transformer.validation_pipeline
        self.meta_valid = meta_valid

    def get_validation_loss(self):
        if self.validate_with_map and self.validation_pipeline is not None:
            return self._get_validation_loss()
        else:
            return super().get_validation_loss()

    def _get_validation_loss(self):
        outputs = self._transform()
        prediction = self._generate_prediction(outputs)
        if len(prediction) == 0:
            return self.validation_loss.setdefault(self.epoch_id, {'sum': Variable(torch.Tensor([0]))})

        with TemporaryDirectory() as temp_dir:
            prediction_filepath = os.path.join(temp_dir, 'prediction.json')
            with open(prediction_filepath, "w") as fp:
                fp.write(json.dumps(prediction))

            annotation_file_path = os.path.join(self.data_dir, 'val', "annotation.json")

            logger.info('Calculating mean precision and recall')
            average_precision, average_recall = coco_evaluation(gt_filepath=annotation_file_path,
                                                                prediction_filepath=prediction_filepath,
                                                                image_ids=self.meta_valid[Y_COLUMNS_SCORING].values,
                                                                category_ids=CATEGORY_IDS[1:],
                                                                small_annotations_size=self.small_annotations_size)
        return self.validation_loss.setdefault(self.epoch_id, {'sum': Variable(torch.Tensor([average_precision]))})

    def _transform(self):
        self.model.eval()
        batch_gen, steps = self.validation_datagen
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
        for name, prediction in outputs.items():
            outputs[name] = softmax(prediction, axis=1)

        return outputs

    def _generate_prediction(self, outputs):
        data = {'input': {'meta': self.meta_valid,
                          'meta_valid': None,
                          'train_mode': False,
                          'target_sizes': [(300, 300)] * len(self.meta_valid),
                          **outputs
                          },
                }

        self.validation_pipeline.clean_cache()
        output = self.validation_pipeline.fit_transform(data)
        self.validation_pipeline.clean_cache()
        y_pred = output['y_pred']

        prediction = create_annotations(self.meta_valid, y_pred, logger, CATEGORY_IDS)
        return prediction
