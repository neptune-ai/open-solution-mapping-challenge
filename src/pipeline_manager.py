import os
import shutil

import pandas as pd
from deepsense import neptune
import crowdai
import json

from .pipeline_config import SOLUTION_CONFIG, Y_COLUMNS_SCORING, CATEGORY_IDS, SEED
from .pipelines import PIPELINES
from .preparation import overlay_masks
from .utils import init_logger, read_params, generate_metadata, set_seed, coco_evaluation, \
    create_annotations, generate_data_frame_chunks


class PipelineManager():
    def __init__(self):
        self.logger = init_logger()
        self.seed = SEED
        set_seed(self.seed)
        self.ctx = neptune.Context()
        self.params = read_params(self.ctx, fallback_file='neptune.yaml')

    def prepare_metadata(self, train_data, valid_data, test_data, public_paths):
        self.logger.info('creating metadata')
        meta = generate_metadata(data_dir=self.params.data_dir,
                                 meta_dir=self.params.meta_dir,
                                 masks_overlayed_dir=self.params.masks_overlayed_dir,
                                 competition_stage=self.params.competition_stage,
                                 process_train_data=train_data,
                                 process_validation_data=valid_data,
                                 process_test_data=test_data,
                                 public_paths=public_paths)

        metadata_filepath = os.path.join(self.params.meta_dir,
                                         'stage{}_metadata.csv').format(self.params.competition_stage)
        self.logger.info('saving metadata to {}'.format(metadata_filepath))
        meta.to_csv(metadata_filepath, index=None)

    def prepare_masks(self, dev_mode):
        for dataset in ["train", "val"]:
            self.logger.info('Overlaying masks, dataset: {}'.format(dataset))
            target_dir = "{}_eroded_{}_dilated_{}".format(self.params.masks_overlayed_dir[:-1],
                                                          self.params.erode_selem_size, self.params.dilate_selem_size)
            self.logger.info('Output directory: {}'.format(target_dir))

            overlay_masks(data_dir=self.params.data_dir,
                          dataset=dataset,
                          target_dir=target_dir,
                          category_ids=CATEGORY_IDS,
                          erode=self.params.erode_selem_size,
                          dilate=self.params.dilate_selem_size,
                          is_small=dev_mode,
                          nthreads=self.params.num_threads,
                          border_width=self.params.border_width,
                          small_annotations_size=self.params.small_annotations_size)

    def train(self, pipeline_name, dev_mode):
        self.logger.info('training')
        if bool(self.params.overwrite) and os.path.isdir(self.params.experiment_dir):
            shutil.rmtree(self.params.experiment_dir)

        meta = pd.read_csv(os.path.join(self.params.meta_dir, 'stage{}_metadata.csv'.format(self.params.competition_stage)),
                           low_memory=False)
        meta_train = meta[meta['is_train'] == 1]
        meta_valid = meta[meta['is_valid'] == 1]

        meta_valid = meta_valid.sample(int(self.params.evaluation_data_sample), random_state=self.seed)

        if dev_mode:
            meta_train = meta_train.sample(20, random_state=self.seed)
            meta_valid = meta_valid.sample(10, random_state=self.seed)

        data = {'input': {'meta': meta_train,
                          'target_sizes': [(300, 300)] * len(meta_train)},
                'specs': {'train_mode': True},
                'callback_input': {'meta_valid': meta_valid}
                }

        pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
        pipeline.clean_cache()
        pipeline.fit_transform(data)
        pipeline.clean_cache()


    def evaluate(self, pipeline_name, dev_mode, chunk_size):
        self.logger.info('evaluating')
        meta = pd.read_csv(os.path.join(self.params.meta_dir,
                                        'stage{}_metadata.csv'.format(self.params.competition_stage)))
        meta_valid = meta[meta['is_valid'] == 1]

        meta_valid = meta_valid.sample(int(self.params.evaluation_data_sample), random_state=self.seed)

        if dev_mode:
            meta_valid = meta_valid.sample(30, random_state=self.seed)

        pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
        prediction = generate_prediction(meta_valid, pipeline, self.logger, CATEGORY_IDS, chunk_size)

        prediction_filepath = os.path.join(self.params.experiment_dir, 'prediction.json')
        with open(prediction_filepath, "w") as fp:
            fp.write(json.dumps(prediction))

        annotation_file_path = os.path.join(self.params.data_dir, 'val', "annotation.json")

        self.logger.info('Calculating mean precision and recall')
        average_precision, average_recall = coco_evaluation(gt_filepath=annotation_file_path,
                                                            prediction_filepath=prediction_filepath,
                                                            image_ids=meta_valid[Y_COLUMNS_SCORING].values,
                                                            category_ids=CATEGORY_IDS[1:],
                                                            small_annotations_size=self.params.small_annotations_size)
        self.logger.info('Mean precision on validation is {}'.format(average_precision))
        self.logger.info('Mean recall on validation is {}'.format(average_recall))
        self.ctx.channel_send('Precision', 0, average_precision)
        self.ctx.channel_send('Recall', 0, average_recall)

    def predict(self, pipeline_name, dev_mode, submit_predictions, chunk_size):
        self.logger.info('predicting')
        meta = pd.read_csv(os.path.join(self.params.meta_dir,
                                        'stage{}_metadata.csv'.format(self.params.competition_stage)))
        meta_test = meta[meta['is_test'] == 1]

        if dev_mode:
            meta_test = meta_test.sample(2, random_state=self.seed)

        pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
        prediction = generate_prediction(meta_test, pipeline, self.logger, CATEGORY_IDS, chunk_size)

        submission = prediction
        submission_filepath = os.path.join(self.params.experiment_dir, 'submission.json')
        with open(submission_filepath, "w") as fp:
            fp.write(json.dumps(submission))
            self.logger.info('submission saved to {}'.format(submission_filepath))
            self.logger.info('submission head \n\n{}'.format(submission[0]))

        if submit_predictions:
            self.make_submission(submission_filepath)

    def make_submission(self, submission_filepath):
        api_key = self.params.api_key

        challenge = crowdai.Challenge("crowdAIMappingChallenge", api_key)
        self.logger.info('submitting predictions to crowdai')
        challenge.submit(submission_filepath)


def generate_prediction(meta_data, pipeline, logger, category_ids, chunk_size):
    if chunk_size is not None:
        return _generate_prediction_in_chunks(meta_data, pipeline, logger, category_ids, chunk_size)
    else:
        return _generate_prediction(meta_data, pipeline, logger, category_ids)


def _generate_prediction(meta_data, pipeline, logger, category_ids):
    data = {'input': {'meta': meta_data,
                      'target_sizes': [(300, 300)] * len(meta_data),
                      },
            'specs': {'train_mode': False},
            'callback_input': {'meta_valid': None}
            }

    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    prediction = create_annotations(meta_data, y_pred, logger, category_ids)
    return prediction


def _generate_prediction_in_chunks(meta_data, pipeline, logger, category_ids, chunk_size):
    prediction = []
    for meta_chunk in generate_data_frame_chunks(meta_data, chunk_size):
        data = {'input': {'meta': meta_chunk,
                          'target_sizes': [(300, 300)] * len(meta_chunk)
                          },
                'specs': {'train_mode': False},
                'callback_input': {'meta_valid': None}
                }

        pipeline.clean_cache()
        output = pipeline.transform(data)
        pipeline.clean_cache()
        y_pred = output['y_pred']

        prediction_chunk = create_annotations(meta_chunk, y_pred, logger, category_ids)
        prediction.extend(prediction_chunk)

    return prediction
