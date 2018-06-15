import os
import shutil

import click
import pandas as pd
from deepsense import neptune
import crowdai
from pycocotools.coco import COCO

from pipeline_config import SOLUTION_CONFIG, Y_COLUMNS_SCORING, CATEGORY_IDS, CATEGORY_LAYERS
from pipelines import PIPELINES
from preparation import overlay_masks
from utils import init_logger, read_params, generate_metadata, set_seed, coco_evaluation, \
    create_annotations, generate_data_frame_chunks
import json

logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx, fallback_file='neptune.yaml')

seed = 1234
set_seed(seed)


@click.group()
def action():
    pass


@action.command()
@click.option('-tr', '--train_data', help='calculate for train data', is_flag=True, required=False)
@click.option('-val', '--valid_data', help='calculate for validation data', is_flag=True, required=False)
@click.option('-te', '--test_data', help='calculate for test data', is_flag=True, required=False)
@click.option('-pub', '--public_paths', help='use public Neptune paths', is_flag=True, required=False)
def prepare_metadata(train_data, valid_data, test_data, public_paths):
    logger.info('creating metadata')
    meta = generate_metadata(data_dir=params.data_dir,
                             meta_dir=params.meta_dir,
                             masks_overlayed_dir=params.masks_overlayed_dir,
                             competition_stage=params.competition_stage,
                             process_train_data=train_data,
                             process_validation_data=valid_data,
                             process_test_data=test_data,
                             public_paths=public_paths)

    metadata_filepath = os.path.join(params.meta_dir, 'stage{}_metadata.csv').format(params.competition_stage)
    logger.info('saving metadata to {}'.format(metadata_filepath))
    meta.to_csv(metadata_filepath, index=None)


@action.command()
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def prepare_masks(dev_mode):
    for dataset in ["train", "val"]:
        logger.info('Overlaying masks, dataset: {}'.format(dataset))
        target_dir = "{}_eroded_{}_dilated_{}".format(params.masks_overlayed_dir[:-1],
                                                      params.erode_selem_size, params.dilate_selem_size)
        logger.info('Output directory: {}'.format(target_dir))

        overlay_masks(data_dir=params.data_dir,
                      dataset=dataset,
                      target_dir=target_dir,
                      category_ids=CATEGORY_IDS,
                      erode=params.erode_selem_size,
                      dilate=params.dilate_selem_size,
                      is_small=dev_mode,
                      nthreads=params.num_threads,
                      border_width=params.border_width,
                      small_annotations_size=params.small_annotations_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train(pipeline_name, dev_mode):
    logger.info('training')
    _train(pipeline_name, dev_mode)


def _train(pipeline_name, dev_mode):
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage{}_metadata.csv'.format(params.competition_stage)),
                       low_memory=False)
    meta_train = meta[meta['is_train'] == 1]
    meta_valid = meta[meta['is_valid'] == 1]

    meta_valid = meta_valid.sample(int(params.evaluation_data_sample), random_state=seed)

    if dev_mode:
        meta_train = meta_train.sample(10000, random_state=seed)
        meta_valid = meta_valid.sample(10, random_state=seed)

    if 'lgbm' in pipeline_name:
        annotations = []
        annotation_file_path = os.path.join(params.data_dir, 'train', "annotation.json")
        coco = COCO(annotation_file_path)
        for image_id in meta['ImageId'].values:
            image_annotations = {}
            for category_id in CATEGORY_IDS:
                annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=category_id)
                category_annotations = coco.loadAnns(annotation_ids)
                image_annotations[category_id] = category_annotations
            annotations.append(image_annotations)
    else:
        annotations = None

    data = {'input': {'meta': meta_train,
                      'target_sizes': [(300, 300)] * len(meta_train),
                      'annotations': annotations},
            'specs': {'train_mode': True,
                      'n_threads': params.num_threads},
            'callback_input': {'meta_valid': meta_valid}
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run evaluation on', type=int, default=None,
              required=False)
def evaluate(pipeline_name, dev_mode, chunk_size):
    logger.info('evaluating')
    _evaluate(pipeline_name, dev_mode, chunk_size)


def _evaluate(pipeline_name, dev_mode, chunk_size):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage{}_metadata.csv'.format(params.competition_stage)))
    meta_valid = meta[meta['is_valid'] == 1]

    meta_valid = meta_valid.sample(int(params.evaluation_data_sample), random_state=seed)

    if dev_mode:
        meta_valid = meta_valid.sample(30, random_state=seed)

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(meta_valid, pipeline, logger, CATEGORY_IDS, CATEGORY_LAYERS, chunk_size)

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
    ctx.channel_send('Precision', 0, average_precision)
    ctx.channel_send('Recall', 0, average_recall)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def predict(pipeline_name, dev_mode, submit_predictions, chunk_size):
    logger.info('predicting')
    _predict(pipeline_name, dev_mode, submit_predictions, chunk_size)


def _predict(pipeline_name, dev_mode, submit_predictions, chunk_size):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage{}_metadata.csv'.format(params.competition_stage)))
    meta_test = meta[meta['is_test'] == 1]

    if dev_mode:
        meta_test = meta_test.sample(2, random_state=seed)

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(meta_test, pipeline, logger, CATEGORY_IDS, CATEGORY_LAYERS, chunk_size)

    submission = prediction
    submission_filepath = os.path.join(params.experiment_dir, 'submission.json')
    with open(submission_filepath, "w") as fp:
        fp.write(json.dumps(submission))
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission[0]))

    if submit_predictions:
        _make_submission(submission_filepath)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run evaluation and prediction on', type=int,
              default=None, required=False)
def train_evaluate_predict(pipeline_name, submit_predictions, dev_mode, chunk_size):
    logger.info('training')
    _train(pipeline_name, dev_mode)
    logger.info('evaluating')
    _evaluate(pipeline_name, dev_mode, chunk_size)
    logger.info('predicting')
    _predict(pipeline_name, dev_mode, submit_predictions, chunk_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run evaluation and prediction on', type=int,
              default=None, required=False)
def train_evaluate(pipeline_name, dev_mode, chunk_size):
    logger.info('training')
    _train(pipeline_name, dev_mode)
    logger.info('evaluating')
    _evaluate(pipeline_name, dev_mode, chunk_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def evaluate_predict(pipeline_name, submit_predictions, dev_mode, chunk_size):
    logger.info('evaluating')
    _evaluate(pipeline_name, dev_mode, chunk_size)
    logger.info('predicting')
    _predict(pipeline_name, dev_mode, submit_predictions, chunk_size)


@action.command()
@click.option('-f', '--submission_filepath', help='filepath to json submission file', required=True)
def submit_predictions(submission_filepath):
    _make_submission(submission_filepath)


def _make_submission(submission_filepath):
    api_key = params.api_key

    challenge = crowdai.Challenge("crowdAIMappingChallenge", api_key)
    logger.info('submitting predictions to crowdai')
    challenge.submit(submission_filepath)


def generate_prediction(meta_data, pipeline, logger, category_ids, category_layers, chunk_size):
    if chunk_size is not None:
        return _generate_prediction_in_chunks(meta_data, pipeline, logger, category_ids, category_layers, chunk_size)
    else:
        return _generate_prediction(meta_data, pipeline, logger, category_ids, category_layers)


def _generate_prediction(meta_data, pipeline, logger, category_ids, category_layers):
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

    prediction = create_annotations(meta_data, y_pred, logger, category_ids, category_layers)
    return prediction


def _generate_prediction_in_chunks(meta_data, pipeline, logger, category_ids, category_layers, chunk_size):
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

        prediction_chunk = create_annotations(meta_chunk, y_pred, logger, category_ids, category_layers)
        prediction.extend(prediction_chunk)

    return prediction


if __name__ == "__main__":
    action()
