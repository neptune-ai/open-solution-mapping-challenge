import os
import shutil
from multiprocessing import set_start_method

set_start_method('spawn')

import click
import pandas as pd
from deepsense import neptune
import crowdai

from pipeline_config import SOLUTION_CONFIG, Y_COLUMNS_SCORING, CATEGORY_IDS
from pipelines import PIPELINES
from preparation import overlay_masks
from utils import init_logger, read_params, create_submission, generate_metadata, set_seed, \
    generate_data_frame_chunks, read_masks
from metrics import mean_precision_and_recall
import json

logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx)

set_seed(1234)


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
def prepare_masks():
    for dataset in ["train", "val"]:
        logger.info('Overlaying masks, dataset: {}'.format(dataset))
        overlay_masks(data_dir=params.data_dir,
                      dataset=dataset,
                      target_dir=params.masks_overlayed_dir,
                      is_small=False)


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

    if dev_mode:
        meta_train = meta_train.sample(500, random_state=1234)
        meta_valid = meta_valid.sample(30, random_state=1234)

    data = {'input': {'meta': meta_train,
                      'meta_valid': meta_valid,
                      'train_mode': True,
                      'target_sizes': [(300, 300)] * len(meta_train),
                      },
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate(pipeline_name, dev_mode):
    logger.info('evaluating')
    _evaluate(pipeline_name, dev_mode)


def _evaluate(pipeline_name, dev_mode):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage{}_metadata.csv'.format(params.competition_stage)))
    meta_valid = meta[meta['is_valid'] == 1]

    if dev_mode:
        meta_valid = meta_valid.sample(30, random_state=1234)

    data = {'input': {'meta': meta_valid,
                      'meta_valid': None,
                      'train_mode': False,
                      'target_sizes': [(300, 300)] * len(meta_valid),
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']
    pipeline.clean_cache()

    y_true = read_masks(meta_valid[Y_COLUMNS_SCORING].values, params.data_dir, dataset="val")

    logger.info('Calculating mean precision and recall')
    (precision, recall) = mean_precision_and_recall(y_true, y_pred)
    logger.info('Mean precision on validation is {}'.format(precision))
    logger.info('Mean recall on validation is {}'.format(recall))
    ctx.channel_send('Precision', 0, precision)
    ctx.channel_send('Recall', 0, recall)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def predict(pipeline_name, dev_mode, submit_predictions, chunk_size):
    logger.info('predicting')
    if chunk_size is not None:
        _predict_in_chunks(pipeline_name, dev_mode, submit_predictions, chunk_size)
    else:
        _predict(pipeline_name, dev_mode, submit_predictions)


def _predict(pipeline_name, dev_mode, submit_predictions):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage{}_metadata.csv'.format(params.competition_stage)))
    meta_test = meta[meta['is_test'] == 1]

    if dev_mode:
        meta_test = meta_test.sample(2, random_state=1234)

    data = {'input': {'meta': meta_test,
                      'meta_valid': None,
                      'train_mode': False,
                      'target_sizes': [(300, 300)] * len(meta_test),
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    submission = create_submission(meta_test, y_pred, logger, CATEGORY_IDS)

    submission_filepath = os.path.join(params.experiment_dir, 'submission.json')
    with open(submission_filepath, "w") as fp:
        fp.write(json.dumps(submission))
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission[0]))

    if submit_predictions:
        _submit_predictions()


def _predict_in_chunks(pipeline_name, submit_predictions, dev_mode, chunk_size):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage{}_metadata.csv'.format(params.competition_stage)))
    meta_test = meta[meta['is_test'] == 1]

    if dev_mode:
        meta_test = meta_test.sample(9, random_state=1234)

    logger.info('processing metadata of shape {}'.format(meta_test.shape))

    submission_chunks = []
    for meta_chunk in generate_data_frame_chunks(meta_test, chunk_size):
        data = {'input': {'meta': meta_chunk,
                          'meta_valid': None,
                          'train_mode': False,
                          'target_sizes': [(300, 300)] * len(meta_chunk)
                          },
                }

        pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
        pipeline.clean_cache()
        output = pipeline.transform(data)
        pipeline.clean_cache()
        y_pred = output['y_pred']

        submission_chunk = create_submission(meta_chunk, y_pred, logger, CATEGORY_IDS)
        submission_chunks.extend(submission_chunk)

    submission_filepath = os.path.join(params.experiment_dir, 'submission.json')
    submission = submission_chunks
    with open(submission_filepath, "w") as fp:
        fp.write(json.dumps(submission))
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission[0]))

    if submit_predictions:
        _submit_predictions()


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def train_evaluate_predict(pipeline_name, submit_predictions, dev_mode, chunk_size):
    logger.info('training')
    _train(pipeline_name, dev_mode)
    logger.info('evaluating')
    _evaluate(pipeline_name, dev_mode)
    logger.info('predicting')
    if chunk_size is not None:
        _predict_in_chunks(pipeline_name, dev_mode, submit_predictions, chunk_size)
    else:
        _predict(pipeline_name, dev_mode, submit_predictions)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate(pipeline_name, dev_mode):
    logger.info('training')
    _train(pipeline_name, dev_mode)
    logger.info('evaluating')
    _evaluate(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def evaluate_predict(pipeline_name, submit_predictions, dev_mode, chunk_size):
    logger.info('evaluating')
    _evaluate(pipeline_name, dev_mode)
    logger.info('predicting')
    if chunk_size is not None:
        _predict_in_chunks(pipeline_name, dev_mode, submit_predictions, chunk_size)
    else:
        _predict(pipeline_name, dev_mode, submit_predictions)


@action.command()
def submit_predictions():
    _submit_predictions()


def _submit_predictions():
    api_key = params.api_key

    challenge = crowdai.Challenge("crowdAIMappingChallenge", api_key)
    submission_filepath = os.path.join(params.experiment_dir, 'submission.json')
    result = challenge.submit(submission_filepath)
    print(result)


if __name__ == "__main__":
    action()
