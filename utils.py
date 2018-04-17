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


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def init_logger():
    logger = logging.getLogger('dsb-2018')
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
    return logging.getLogger('dsb-2018')


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


def create_submission(meta, predictions, logger):
    return NotImplementedError


def read_masks(masks_filepaths):
    masks = []
    for mask_dir in tqdm(masks_filepaths):
        mask = []
        if len(mask_dir) == 1:
            mask_dir = mask_dir[0]
        for i, mask_filepath in enumerate(glob.glob('{}/*'.format(mask_dir))):
            blob = np.asarray(Image.open(mask_filepath))
            blob_binarized = (blob > 128.).astype(np.uint8) * i
            mask.append(blob_binarized)
        mask = np.sum(np.stack(mask, axis=0), axis=0).astype(np.uint8)
        masks.append(mask)
    return masks


def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1] + rle[-2] == x.size:
        rle[-2] = rle[-2] - 1

    return rle


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml('neptune.yaml')
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def generate_metadata(data_dir,
                      masks_overlayed_dir,
                      contours_overlayed_dir,
                      centers_overlayed_dir,
                      competition_stage=1,
                      process_train_data=True,
                      process_test_data=True):
    def _generate_metadata(train):
        df_metadata = pd.DataFrame(columns=['ImageId', 'file_path_image', 'file_path_masks', 'file_path_mask',
                                            'is_train', 'width', 'height', 'n_nuclei'])
        if train:
            tr_te = 'stage{}_train'.format(competition_stage)
        else:
            tr_te = 'stage{}_test'.format(competition_stage)

        for image_id in sorted(os.listdir(os.path.join(data_dir, tr_te))):
            p = os.path.join(data_dir, tr_te, image_id, 'images')
            if image_id != os.listdir(p)[0][:-4]:
                ValueError('ImageId mismatch ' + str(image_id))
            if len(os.listdir(p)) != 1:
                ValueError('more than one image in dir')

            file_path_image = os.path.join(p, os.listdir(p)[0])
            if train:
                is_train = 1
                file_path_masks = os.path.join(data_dir, tr_te, image_id, 'masks')
                file_path_mask = os.path.join(masks_overlayed_dir, tr_te, image_id + '.png')
                file_path_contours = os.path.join(contours_overlayed_dir, tr_te, image_id + '.png')
                file_path_centers = os.path.join(centers_overlayed_dir, tr_te, image_id + '.png')
                n_nuclei = len(os.listdir(file_path_masks))
            else:
                is_train = 0
                file_path_masks = None
                file_path_mask = None
                file_path_contours = None
                file_path_contours_touching = None
                file_path_centers = None
                n_nuclei = None

            img = Image.open(file_path_image)
            width = img.size[0]
            height = img.size[1]
            s = df_metadata['ImageId']
            if image_id is s:
                ValueError('ImageId conflict ' + str(image_id))
            df_metadata = df_metadata.append({'ImageId': image_id,
                                              'file_path_image': file_path_image,
                                              'file_path_masks': file_path_masks,
                                              'file_path_mask': file_path_mask,
                                              'file_path_contours': file_path_contours,
                                              'file_path_centers': file_path_centers,
                                              'is_train': is_train,
                                              'width': width,
                                              'height': height,
                                              'n_nuclei': n_nuclei}, ignore_index=True)
        return df_metadata

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


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


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
