import json
import logging
import math
import os
import random
import sys
from itertools import product

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from attrdict import AttrDict
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def read_masks(image_ids, data_dir, dataset):
    masks = []
    annotation_file_name = "annotation.json"
    annotation_file_path = os.path.join(data_dir, dataset, annotation_file_name)
    coco = COCO(annotation_file_path)
    for image_id in tqdm(image_ids):
        mask_set = []
        image = coco.loadImgs(image_id)[0]
        image_size = [image["height"], image["width"]]
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)
        for ann in annotations:
            rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
            m = cocomask.decode(rle)
            mask_set.append(m)
        masks.append(mask_set)
    return masks


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


def create_annotations(meta, predictions, scores, logger, category_ids, save=False, experiment_dir='./'):
    '''
    :param meta: pd.DataFrame with metadata
    :param predictions: list of labeled masks or numpy array of size [n_images, im_height, im_width]
    :param logger:
    :param save: True, if one want to save submission, False if one want to return it
    :param experiment_dir: path to save submission
    :return: submission if save==False else True
    '''
    annotations = []
    logger.info('Creating annotations')
    for image_id, prediction, image_scores in zip(meta["ImageId"].values, predictions, scores):
        for category_nr, (category_instances, category_scores) in enumerate(zip(prediction, image_scores)):
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


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml('neptune.yaml')
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def generate_metadata(data_dir,
                      masks_overlayed_dir,
                      masks_overlayed_eroded_dir,
                      process_train_data=True,
                      process_validation_data=True,
                      process_test_data=True,
                      public_paths=False,
                      competition_stage=1):
    if competition_stage != 1:
        raise NotImplementedError('only stage_1 is supported for now')

    def _generate_metadata(dataset):
        assert dataset in ["train", "test", "val"], "Unknown dataset!"
        df_metadata = pd.DataFrame(columns=['ImageId', 'file_path_image', 'file_path_mask', 'file_path_mask_eroded',
                                            'is_train', 'is_valid', 'is_test', 'n_buildings'])

        if dataset == "test":
            dataset = "test_images"

        images_path = os.path.join(data_dir, dataset)
        public_path = "/public/mapping_challenge_data/"

        if dataset != "test_images":
            images_path = os.path.join(images_path, "images")

        if public_paths:
            images_path_to_write = os.path.join(public_path, dataset)
            mask_overlayed_suffix = os.path.join(masks_overlayed_dir, "")
            masks_overlayed_dir_to_write = os.path.join(public_path, mask_overlayed_suffix.split("/")[-2])
            mask_overlayed_eroded_suffix = os.path.join(masks_overlayed_eroded_dir, "")
            masks_overlayed_eroded_dir_to_write = os.path.join(public_path,
                                                               mask_overlayed_eroded_suffix.split("/")[:-2])
        else:
            images_path_to_write = images_path
            masks_overlayed_dir_to_write = masks_overlayed_dir
            masks_overlayed_eroded_dir_to_write = masks_overlayed_eroded_dir

        for image_file_name in sorted(os.listdir(images_path)):
            file_path_image = os.path.join(images_path_to_write, image_file_name)
            image_id = image_file_name[:-4]

            is_train = 0
            is_valid = 0
            is_test = 0

            if dataset == "test_images":
                file_path_mask = None
                file_path_mask_eroded = None
                n_buildings = None
                is_test = 1
            else:
                file_path_mask = os.path.join(masks_overlayed_dir_to_write, dataset, "masks",
                                              image_file_name[:-4] + ".png")
                file_path_mask_eroded = os.path.join(masks_overlayed_eroded_dir_to_write, dataset, "masks",
                                                     image_file_name[:-4] + ".png")
                n_buildings = None
                if dataset == "val":
                    is_valid = 1
                else:
                    is_train = 1

            df_metadata = df_metadata.append({'ImageId': image_id,
                                              'file_path_image': file_path_image,
                                              'file_path_mask': file_path_mask,
                                              'file_path_mask_eroded': file_path_mask_eroded,
                                              'is_train': is_train,
                                              'is_valid': is_valid,
                                              'is_test': is_test,
                                              'n_buildings': n_buildings}, ignore_index=True)

        return df_metadata

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
    for i in tqdm(range(chunk_nr)):
        meta_chunk = meta.iloc[i * chunk_size:(i + 1) * chunk_size]
        yield meta_chunk


def categorize_image(image, channel_axis=0):
    return np.argmax(image, axis=channel_axis)


def coco_evaluation(gt_filepath, prediction_filepath, image_ids, category_ids):
    coco = COCO(gt_filepath)
    coco_results = coco.loadRes(prediction_filepath)
    cocoEval = COCOeval(coco, coco_results)
    cocoEval.params.imgIds = image_ids
    cocoEval.params.catIds = category_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats[1], cocoEval.stats[8]


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled


def get_weight_matrix(mask):
    labeled = label(mask)
    image_size = mask.shape
    distances = np.zeros(image_size)
    for label_nr in range(1, labeled.max() + 1):
        object = labeled == label_nr
        if distances.sum() == 0:
            distances = distance_transform_edt(1 - object)
        else:
            distances = np.dstack([distances, distance_transform_edt(1 - object)])
    if np.sum(distances) != 0:
        if len(distances.shape) > 2:
            distances.sort(axis=2)
            weights = get_weights(distances[:, :, 0], distances[:, :, 1], 1, 10, 5)
        else:
            weights = get_weights(0, distances, 1, 10, 5)
    else:
        weights = np.ones(image_size)
    return weights


def get_weights(d1, d2, w1, w0, sigma):
    return w1 + w0 * np.exp(-((d1 + d2) ** 2) / (sigma ** 2))


def denormalize_img(image, mean, std):
    return image * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled


def add_dropped_objects(original, processed):
    reconstructed = processed.copy()
    labeled = label(original)
    for i in range(1, labeled.max() + 1):
        if np.any(np.where(~(labeled == i) & processed)):
            reconstructed += (labeled == i)
    return reconstructed.astype('uint8')
