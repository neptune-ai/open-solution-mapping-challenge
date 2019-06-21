import json
import logging
import math
import os
import ntpath
import random
import sys
import time
from itertools import product, chain
from collections import defaultdict, Iterable

import glob
import numpy as np
import pandas as pd
import torch
import yaml
import imgaug as ia
from PIL import Image
from attrdict import AttrDict
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from tqdm import tqdm
from scipy import ndimage as ndi
from .cocoeval import COCOeval
from .steps.base import BaseTransformer


def init_logger():
    logger = logging.getLogger('mapping-challenge')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def get_logger():
    return logging.getLogger('mapping-challenge')


def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)

    if not masks:
        return [labeled]
    else:
        return masks


def create_annotations(meta, predictions, logger, category_ids, category_layers, save=False, experiment_dir='./'):
    """

    Args:
        meta: pd.DataFrame with metadata
        predictions: list of labeled masks or numpy array of size [n_images, im_height, im_width]
        logger: logging object
        category_ids: list with ids of categories,
            e.g. [None, 100] means, that no annotations will be created from category 0 data, and annotations
            from category 1 will be created with category_id=100
        category_layers:
        save: True, if one want to save submission, False if one want to return it
        experiment_dir: directory of experiment to save annotations, relevant if save==True

    Returns: submission if save==False else True

    """
    annotations = []
    logger.info('Creating annotations')
    category_layers_inds = np.cumsum(category_layers)
    for image_id, (prediction, image_scores) in zip(meta["ImageId"].values, predictions):
        for category_ind, (category_instances, category_scores) in enumerate(zip(prediction, image_scores)):
            category_nr = np.searchsorted(category_layers_inds, category_ind, side='right')
            if category_ids[category_nr] != None:
                masks = decompose(category_instances)
                for mask_nr, (mask, score) in enumerate(zip(masks, category_scores)):
                    annotation = {}
                    annotation["image_id"] = int(image_id)
                    annotation["category_id"] = category_ids[category_nr]
                    annotation["score"] = score
                    annotation["segmentation"] = rle_from_binary(mask.astype('uint8'))
                    annotation['segmentation']['counts'] = annotation['segmentation']['counts'].decode("UTF-8")
                    annotation["bbox"] = bounding_box_from_rle(rle_from_binary(mask.astype('uint8')))
                    annotations.append(annotation)
    if save:
        submission_filepath = os.path.join(experiment_dir, 'submission.json')
        with open(submission_filepath, "w") as fp:
            fp.write(str(json.dumps(annotations)))
            logger.info("Submission saved to {}".format(submission_filepath))
            logger.info('submission head \n\n{}'.format(annotations[0]))
        return True
    else:
        return annotations


def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)


def bounding_box_from_rle(rle):
    return list(cocomask.toBbox(rle))


def read_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f)
    return AttrDict(config)


def generate_metadata(data_dir,
                      meta_dir,
                      masks_overlayed_prefix,
                      process_train_data=True,
                      process_validation_data=True,
                      process_test_data=True,
                      public_paths=False,
                      competition_stage=1,
                      ):
    if competition_stage != 1:
        raise NotImplementedError('only stage_1 is supported for now')

    def _generate_metadata(dataset):
        assert dataset in ["train", "test", "val"], "Unknown dataset!"

        if dataset == "test":
            dataset = "test_images"

        images_path = os.path.join(data_dir, dataset)

        if dataset != "test_images":
            images_path = os.path.join(images_path, "images")

        if public_paths:
            raise NotImplementedError('public neptune paths not implemented')
        else:
            masks_overlayed_dirs, mask_overlayed_suffix = [], []
            for file_path in glob.glob('{}/*'.format(meta_dir)):
                if ntpath.basename(file_path).startswith(masks_overlayed_prefix):
                    masks_overlayed_dirs.append(file_path)
                    mask_overlayed_suffix.append(ntpath.basename(file_path).replace(masks_overlayed_prefix, ''))
        df_dict = defaultdict(lambda: [])

        for image_file_path in tqdm(sorted(glob.glob('{}/*'.format(images_path)))):
            image_id = ntpath.basename(image_file_path).split('.')[0]

            is_train = 0
            is_valid = 0
            is_test = 0

            if dataset == "test_images":
                n_buildings = None
                is_test = 1
                df_dict['ImageId'].append(image_id)
                df_dict['file_path_image'].append(image_file_path)
                df_dict['is_train'].append(is_train)
                df_dict['is_valid'].append(is_valid)
                df_dict['is_test'].append(is_test)
                df_dict['n_buildings'].append(n_buildings)
                for mask_dir_suffix in mask_overlayed_suffix:
                    df_dict['file_path_mask' + mask_dir_suffix].append(None)

            else:
                n_buildings = None
                if dataset == "val":
                    is_valid = 1
                else:
                    is_train = 1
                df_dict['ImageId'].append(image_id)
                df_dict['file_path_image'].append(image_file_path)
                df_dict['is_train'].append(is_train)
                df_dict['is_valid'].append(is_valid)
                df_dict['is_test'].append(is_test)
                df_dict['n_buildings'].append(n_buildings)

                for mask_dir, mask_dir_suffix in zip(masks_overlayed_dirs, mask_overlayed_suffix):
                    file_path_mask = os.path.join(mask_dir, dataset, "masks", '{}.png'.format(image_id))
                    df_dict['file_path_mask' + mask_dir_suffix].append(file_path_mask)

        return pd.DataFrame.from_dict(df_dict)

    metadata = pd.DataFrame()
    if process_train_data:
        train_metadata = _generate_metadata(dataset="train")
        metadata = metadata.append(train_metadata, ignore_index=True)
    if process_validation_data:
        validation_metadata = _generate_metadata(dataset="val")
        metadata = metadata.append(validation_metadata, ignore_index=True)
    if process_test_data:
        test_metadata = _generate_metadata(dataset="test")
        metadata = metadata.append(test_metadata, ignore_index=True)
    if not (process_test_data or process_train_data or process_validation_data):
        raise ValueError('At least one of train_data, validation_data or test_data has to be set to True')

    return metadata


def squeeze_inputs(inputs):
    return np.squeeze(inputs[0], axis=1)


def softmax(X, theta=1.0, axis=None):
    """
    https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def from_pil(*images):
    images = [np.array(image) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def to_pil(*images):
    images = [Image.fromarray((image).astype(np.uint8)) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_data_frame_chunks(meta, chunk_size):
    n_rows = meta.shape[0]
    chunk_nr = math.ceil(n_rows / chunk_size)
    for i in tqdm(range(chunk_nr)):
        meta_chunk = meta.iloc[i * chunk_size:(i + 1) * chunk_size]
        yield meta_chunk


def coco_evaluation(gt_filepath, prediction_filepath, image_ids, category_ids, small_annotations_size):
    coco = COCO(gt_filepath)
    coco_results = coco.loadRes(prediction_filepath)
    cocoEval = COCOeval(coco, coco_results)
    cocoEval.params.imgIds = image_ids
    cocoEval.params.catIds = category_ids
    cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, small_annotations_size ** 2],
                               [small_annotations_size ** 2, 1e5 ** 2]]
    cocoEval.params.areaRngLbl = ['all', 'small', 'large']
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats[0], cocoEval.stats[3]


def denormalize_img(image, mean, std):
    return image * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled


def add_dropped_objects(original, processed):
    reconstructed = processed.copy()
    labeled = label(original)
    for i in range(1, labeled.max() + 1):
        if not np.any(np.where((labeled == i) & processed)):
            reconstructed += (labeled == i)
    return reconstructed.astype('uint8')


def make_apply_transformer(func, output_name='output', apply_on=None):
    class StaticApplyTransformer(BaseTransformer):
        def transform(self, *args, **kwargs):
            self.check_input(*args, **kwargs)

            if not apply_on:
                iterator = zip(*args, *kwargs.values())
            else:
                iterator = zip(*args, *[kwargs[key] for key in apply_on])

            output = []
            for func_args in tqdm(iterator, total=self.get_arg_length(*args, **kwargs)):
                output.append(func(*func_args))
            return {output_name: output}

        @staticmethod
        def check_input(*args, **kwargs):
            if len(args) and len(kwargs) == 0:
                raise Exception('Input must not be empty')

            arg_length = None
            for arg in chain(args, kwargs.values()):
                if not isinstance(arg, Iterable):
                    raise Exception('All inputs must be iterable')
                arg_length_loc = None
                try:
                    arg_length_loc = len(arg)
                except:
                    pass
                if arg_length_loc is not None:
                    if arg_length is None:
                        arg_length = arg_length_loc
                    elif arg_length_loc != arg_length:
                        raise Exception('All inputs must be the same length')

        @staticmethod
        def get_arg_length(*args, **kwargs):
            arg_length = None
            for arg in chain(args, kwargs.values()):
                if arg_length is None:
                    try:
                        arg_length = len(arg)
                    except:
                        pass
                if arg_length is not None:
                    return arg_length

    return StaticApplyTransformer()


def make_apply_transformer_stream(func, output_name='output', apply_on=None):
    class StaticApplyTransformerStream(BaseTransformer):
        def transform(self, *args, **kwargs):
            self.check_input(*args, **kwargs)
            return {output_name: self._transform(*args, **kwargs)}

        def _transform(self, *args, **kwargs):
            if not apply_on:
                iterator = zip(*args, *kwargs.values())
            else:
                iterator = zip(*args, *[kwargs[key] for key in apply_on])

            for func_args in tqdm(iterator):
                yield func(*func_args)

        @staticmethod
        def check_input(*args, **kwargs):
            for arg in chain(args, kwargs.values()):
                if not isinstance(arg, Iterable):
                    raise Exception('All inputs must be iterable')

    return StaticApplyTransformerStream()


def get_seed():
    seed = int(time.time()) + int(os.getpid())
    return seed


def reseed(augmenter_sequence, deterministic=True):
    for aug in augmenter_sequence:
        aug.random_state = ia.new_random_state(get_seed())
        if deterministic:
            aug.deterministic = True
    return augmenter_sequence
