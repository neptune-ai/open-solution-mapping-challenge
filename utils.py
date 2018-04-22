import glob
import logging
import os
import random
import sys
from itertools import product
import math

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from attrdict import AttrDict
from tqdm import tqdm
import json
import pycocotools


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


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


def create_submission(meta, predictions, logger, save=False, experiment_dir='./'):
    '''
    :param meta: pd.DataFrame with metadata
    :param predictions: list of labeled masks or numpy array of size [n_images, im_height, im_width]
    :param logger:
    :param save: True, if one want to save submission, False if one want to return it
    :param experiment_dir: path to save submission
    :return: submission if save==False else True
    '''
    annotations = []
    for image_id, prediction in zip(meta["ImageId"].values, predictions):
        score = 1.0
        masks = decompose(prediction)
        for mask_nr, mask in enumerate(masks):
            annotation = {}
            annotation["image_id"] = image_id
            annotation["category_id"] = 100
            annotation["score"] = score
            annotation["segmentation"] = rle_from_binary(mask)
            annotation["bbox"] = bounding_box_from_rle(annotation["segmentation"])
            annotations.append(annotation)
    if save:
        submission_filepath = os.path.join(experiment_dir, 'submission.json')
        with open(submission_filepath, "w") as fp:
            fp.write(json.dumps(annotations))
            logger.info("Submission saved to {}".format(submission_filepath))
        return True
    else:
        return annotations


def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return pycocotools.mask.encode(prediction)


def bounding_box_from_rle(rle):
    return pycocotools.toBbox(rle)


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml('neptune.yaml')
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def generate_metadata(data_dir,
                      process_train_data=True,
                      process_test_data=True):
    def _generate_metadata(train):
        pass

    if process_train_data and process_test_data:
        train_metadata = _generate_metadata(train=True)
        test_metadata = _generate_metadata(train=False)
        metadata = train_metadata.append(test_metadata, ignore_index=True)
    elif process_train_data and not process_test_data:
        metadata = _generate_metadata(train=True)
    elif not process_train_data and process_test_data:
        metadata = _generate_metadata(train=False)
    else:
        raise ValueError('both train_data and test_data cannot be set to False')

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


def relabel(img):
    h, w = img.shape

    relabel_dict = {}

    for i, k in enumerate(np.unique(img)):
        if k == 0:
            relabel_dict[k] = 0
        else:
            relabel_dict[k] = i
    for i, j in product(range(h), range(w)):
        img[i, j] = relabel_dict[img[i, j]]
    return img


def relabel_random_colors(img, max_colours=1000):
    keys = list(range(1, max_colours, 1))
    np.random.shuffle(keys)
    values = list(range(1, max_colours, 1))
    np.random.shuffle(values)
    funky_dict = {k: v for k, v in zip(keys, values)}
    funky_dict[0] = 0

    h, w = img.shape

    for i, j in product(range(h), range(w)):
        img[i, j] = funky_dict[img[i, j]]
    return img


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


def clip(lo, x, hi):
    return lo if x <= lo else hi if x >= hi else x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_data_frame_chunks(meta, chunk_size):
    n_rows = meta.shape[0]
    chunk_nr = math.ceil(n_rows / chunk_size)
    meta_chunks = []
    for i in tqdm(range(chunk_nr)):
        meta_chunk = meta.iloc[i * chunk_size:(i + 1) * chunk_size]
        yield meta_chunk
