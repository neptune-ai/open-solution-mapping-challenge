import os
import shutil

import pandas as pd
import neptune
import json
from pycocotools.coco import COCO

from .pipeline_config import SOLUTION_CONFIG, Y_COLUMNS_SCORING, CATEGORY_IDS, SEED, CATEGORY_LAYERS
from .pipelines import PIPELINES
from .preparation import overlay_masks
from .utils import init_logger, read_config, get_filepaths, generate_metadata, set_seed, coco_evaluation, \
    create_annotations, generate_data_frame_chunks, generate_inference_metadata


class PipelineManager:
    def __init__(self):
        self.logger = init_logger()
        self.seed = SEED
        set_seed(self.seed)
        self.config = read_config(config_path=os.getenv('CONFIG_PATH'))
        self.params = self.config.parameters

    def start_experiment(self):
        neptune.init(project_qualified_name=self.config.project)
        neptune.create_experiment(name=self.config.name,
                                  params=self.params,
                                  upload_source_files=get_filepaths(),
                                  tags=self.config.tags)

    def prepare_masks(self, dev_mode):
        prepare_masks(dev_mode, self.logger, self.params)

    def prepare_metadata(self, train_data, valid_data, test_data, public_paths):
        prepare_metadata(train_data, valid_data, test_data, public_paths, self.logger, self.params)

    def train(self, pipeline_name, dev_mode):
        if 'scoring' in pipeline_name:
            assert CATEGORY_LAYERS[1] > 1, """You are running training on a second layer model that chooses 
               which threshold should be chosen for a particular image. You need to specify a larger number of 
               possible thresholds in the CATEGORY_LAYERS, suggested is 19"""
        train(pipeline_name, dev_mode, self.logger, self.params, self.seed)

    def evaluate(self, pipeline_name, dev_mode, chunk_size):
        if 'scoring' in pipeline_name:
            assert CATEGORY_LAYERS[1] > 1, """You are running inference with a second layer model that chooses 
               which threshold should be chosen for a particular image. You need to specify a larger number of 
               possible thresholds in the CATEGORY_LAYERS, suggested is 19"""
        else:
            assert CATEGORY_LAYERS[1] == 1, """You are running inference without a second layer model.
            Change thresholds setup in CATEGORY_LAYERS to [1,1]"""
        evaluate(pipeline_name, dev_mode, chunk_size, self.logger, self.params, self.seed)

    def predict(self, pipeline_name, dev_mode, submit_predictions, chunk_size):
        if 'scoring' in pipeline_name:
            assert CATEGORY_LAYERS[1] > 1, """You are running inference with a second layer model that chooses 
               which threshold should be chosen for a particular image. You need to specify a larger number of 
               possible thresholds in the CATEGORY_LAYERS, suggested is 19"""
        else:
            assert CATEGORY_LAYERS[1] == 1, """You are running inference without a second layer model.
            Change thresholds setup in CATEGORY_LAYERS to [1,1]"""
        predict(pipeline_name, dev_mode, submit_predictions, chunk_size, self.logger, self.params, self.seed)

    def predict_on_dir(self, pipeline_name, dir_path, prediction_path, chunk_size):
        if 'scoring' in pipeline_name:
            assert CATEGORY_LAYERS[1] > 1, """You are running inference with a second layer model that chooses 
               which threshold should be chosen for a particular image. You need to specify a larger number of 
               possible thresholds in the CATEGORY_LAYERS, suggested is 19"""
        else:
            assert CATEGORY_LAYERS[1] == 1, """You are running inference without a second layer model.
            Change thresholds setup in CATEGORY_LAYERS to [1,1]"""
        predict_on_dir(pipeline_name, dir_path, prediction_path, chunk_size, self.logger, self.params)

    def make_submission(self, submission_filepath):
        make_submission(submission_filepath, self.logger, self.params)

    def finish_experiment(self):
        neptune.stop()


def prepare_masks(dev_mode, logger, params):
    for dataset in ["train", "val"]:
        logger.info('Overlaying masks, dataset: {}'.format(dataset))

        mask_dirname = "masks_overlayed_eroded_{}_dilated_{}".format(params.erode_selem_size, params.dilate_selem_size)
        target_dir = os.path.join(params.meta_dir, mask_dirname)
        logger.info('Output directory: {}'.format(target_dir))

        overlay_masks(data_dir=params.data_dir,
                      dataset=dataset,
                      target_dir=target_dir,
                      category_ids=CATEGORY_IDS,
                      erode=params.erode_selem_size,
                      dilate=params.dilate_selem_size,
                      is_small=dev_mode,
                      num_threads=params.num_threads,
                      border_width=params.border_width,
                      small_annotations_size=params.small_annotations_size)


def prepare_metadata(train_data, valid_data, test_data, public_paths, logger, params):
    logger.info('creating metadata')

    meta = generate_metadata(data_dir=params.data_dir,
                             meta_dir=params.meta_dir,
                             masks_overlayed_prefix=params.masks_overlayed_prefix,
                             competition_stage=params.competition_stage,
                             process_train_data=train_data,
                             process_validation_data=valid_data,
                             process_test_data=test_data,
                             public_paths=public_paths)

    metadata_filepath = os.path.join(params.meta_dir, 'stage{}_metadata.csv').format(params.competition_stage)
    logger.info('saving metadata to {}'.format(metadata_filepath))
    meta.to_csv(metadata_filepath, index=None)


def train(pipeline_name, dev_mode, logger, params, seed):
    logger.info('training')
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage{}_metadata.csv'.format(params.competition_stage)),
                       low_memory=False)
    meta_train = meta[meta['is_train'] == 1]
    meta_valid = meta[meta['is_valid'] == 1]

    train_mode = True

    meta_valid = meta_valid.sample(int(params.evaluation_data_sample), random_state=seed)

    if dev_mode:
        meta_train = meta_train.sample(20, random_state=seed)
        meta_valid = meta_valid.sample(10, random_state=seed)

    if pipeline_name == 'scoring_model':
        train_mode = False
        meta_train, annotations = _get_scoring_model_data(params.data_dir, meta_train,
                                                          params.scoring_model__num_training_examples, seed)
    else:
        annotations = None

    data = {'input': {'meta': meta_train,
                      'target_sizes': [(300, 300)] * len(meta_train),
                      'annotations': annotations},
            'specs': {'train_mode': train_mode,
                      'num_threads': params.num_threads},
            'callback_input': {'meta_valid': meta_valid}
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


def evaluate(pipeline_name, dev_mode, chunk_size, logger, params, seed):
    logger.info('evaluating')
    meta = pd.read_csv(os.path.join(params.meta_dir,
                                    'stage{}_metadata.csv'.format(params.competition_stage)))
    meta_valid = meta[meta['is_valid'] == 1]

    meta_valid = meta_valid.sample(int(params.evaluation_data_sample), random_state=seed)

    if dev_mode:
        meta_valid = meta_valid.sample(30, random_state=seed)

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(meta_valid, pipeline, logger, CATEGORY_IDS, chunk_size, params.num_threads)

    prediction_filepath = os.path.join(params.experiment_dir, 'prediction.json')
    with open(prediction_filepath, "w") as fp:
        fp.write(json.dumps(prediction))

    annotation_file_path = os.path.join(params.data_dir, 'val', "annotation.json")

    logger.info('Calculating mean precision and recall')
    average_precision, average_recall = coco_evaluation(gt_filepath=annotation_file_path,
                                                        prediction_filepath=prediction_filepath,
                                                        image_ids=meta_valid[Y_COLUMNS_SCORING].values,
                                                        category_ids=CATEGORY_IDS[1:],
                                                        small_annotations_size=params.small_annotations_size)
    logger.info('Mean precision on validation is {}'.format(average_precision))
    logger.info('Mean recall on validation is {}'.format(average_recall))
    neptune.send_metric('Precision', average_precision)
    neptune.send_metric('Recall', average_recall)


def predict(pipeline_name, dev_mode, submit_predictions, chunk_size, logger, params, seed):
    logger.info('predicting')
    meta = pd.read_csv(os.path.join(params.meta_dir,
                                    'stage{}_metadata.csv'.format(params.competition_stage)))
    meta_test = meta[meta['is_test'] == 1]

    if dev_mode:
        meta_test = meta_test.sample(2, random_state=seed)

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(meta_test, pipeline, logger, CATEGORY_IDS, chunk_size, params.num_threads)

    submission = prediction
    submission_filepath = os.path.join(params.experiment_dir, 'submission.json')
    with open(submission_filepath, "w") as fp:
        fp.write(json.dumps(submission))
        logger.info('submission saved to {}'.format(submission_filepath))
        logger.info('submission head \n\n{}'.format(submission[0]))


def predict_on_dir(pipeline_name, dir_path, prediction_path, chunk_size, logger, params):
    logger.info('creating metadata')
    meta = generate_inference_metadata(images_dir=dir_path)

    logger.info('predicting')
    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(meta, pipeline, logger, CATEGORY_IDS, chunk_size, params.num_threads)

    with open(prediction_path, "w") as fp:
        fp.write(json.dumps(prediction))
        logger.info('submission saved to {}'.format(prediction_path))
        logger.info('submission head \n\n{}'.format(prediction[0]))


def generate_prediction(meta_data, pipeline, logger, category_ids, chunk_size, num_threads=1):
    if chunk_size is not None:
        return _generate_prediction_in_chunks(meta_data, pipeline, logger, category_ids, chunk_size, num_threads)
    else:
        return _generate_prediction(meta_data, pipeline, logger, category_ids, num_threads)


def _generate_prediction(meta_data, pipeline, logger, category_ids, num_threads=1):
    data = {'input': {'meta': meta_data,
                      'target_sizes': [(300, 300)] * len(meta_data),
                      },
            'specs': {'train_mode': False,
                      'num_threads': num_threads},
            'callback_input': {'meta_valid': None}
            }

    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    prediction = create_annotations(meta_data, y_pred, logger, category_ids, CATEGORY_LAYERS)
    return prediction


def _generate_prediction_in_chunks(meta_data, pipeline, logger, category_ids, chunk_size, num_threads=1):
    prediction = []
    for meta_chunk in generate_data_frame_chunks(meta_data, chunk_size):
        data = {'input': {'meta': meta_chunk,
                          'target_sizes': [(300, 300)] * len(meta_chunk)
                          },
                'specs': {'train_mode': False,
                          'num_threads': num_threads},
                'callback_input': {'meta_valid': None}
                }
        pipeline.clean_cache()
        output = pipeline.transform(data)
        pipeline.clean_cache()
        y_pred = output['y_pred']

        prediction_chunk = create_annotations(meta_chunk, y_pred, logger, category_ids, CATEGORY_LAYERS)
        prediction.extend(prediction_chunk)

    return prediction


def _get_scoring_model_data(data_dir, meta, num_training_examples, random_seed):
    annotation_file_path = os.path.join(data_dir, 'train', "annotation.json")
    coco = COCO(annotation_file_path)
    meta = meta.sample(num_training_examples, random_state=random_seed)
    annotations = []
    for image_id in meta['ImageId'].values:
        image_annotations = {}
        for category_id in CATEGORY_IDS:
            annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=category_id)
            category_annotations = coco.loadAnns(annotation_ids)
            image_annotations[category_id] = category_annotations
        annotations.append(image_annotations)
    return meta, annotations
