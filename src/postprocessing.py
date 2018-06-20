import multiprocessing as mp

import numpy as np
from skimage.transform import resize
from skimage.morphology import erosion, dilation, rectangle
from tqdm import tqdm
from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax
from pycocotools import mask as cocomask
import pandas as pd
import cv2

from .steps.base import BaseTransformer
from .utils import denormalize_img, add_dropped_objects, label, rle_from_binary
from .pipeline_config import MEAN, STD, CATEGORY_LAYERS, CATEGORY_IDS


class FeatureExtractor(BaseTransformer):
    def __init__(self, n_threads=1):
        self.n_threads = n_threads

    def transform(self, images, probabilities, annotations=None):
        if annotations is None:
            annotations = [{}] * len(images)
        all_features = []
        for image, im_probabilities, im_annotations in zip(images, probabilities, annotations):
            all_features.append(get_features_for_image(image, im_probabilities, im_annotations))
        return {'features': all_features}


class ScoreImageJoiner(BaseTransformer):
    def transform(self, images, scores):
        images_with_scores = []
        for image, score in tqdm(zip(images, scores)):
            images_with_scores.append((image, score))
        return {'images_with_scores': images_with_scores}


class NonMaximumSupression(BaseTransformer):
    def __init__(self, iou_threshold, n_threads=1):
        self.iou_threshold = iou_threshold
        self.n_threads = n_threads

    def transform(self, images_with_scores):
        with mp.pool.ThreadPool(self.n_threads) as executor:
            cleaned_images_with_scores = executor.map(
                lambda p: remove_overlapping_masks(*p, iou_threshold=self.iou_threshold), images_with_scores)
        return {'images_with_scores': cleaned_images_with_scores}


def resize_image(image, target_size):
    """Resize image to target size

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Resized image of shape (C x H x W).

    """
    n_channels = image.shape[0]
    resized_image = resize(image, (n_channels,) + target_size, mode='constant')
    return resized_image


def categorize_image(image):
    """Maps probability map to categories. Each pixel is assigned with a category with highest probability.

    Args:
        image (numpy.ndarray): Probability map of shape (C x H x W).

    Returns:
        numpy.ndarray: Categorized image of shape (H x W).

    """
    return np.argmax(image, axis=0)


def categorize_multilayer_image(image):
    categorized_image = []
    for category_id, category_output in enumerate(image):
        thrs_step = 1. / (CATEGORY_LAYERS[category_id] + 1)
        thresholds = np.arange(thrs_step, 1, thrs_step)
        for thrs in thresholds:
            categorized_image.append(category_output > thrs)
    return np.stack(categorized_image)


def label_multiclass_image(mask):
    """Label separate class instances on a mask.

    Input mask is a 2D numpy.ndarray, cell (h, w) contains class number of that cell.
    Class number has to be an integer from 0 to C - 1, where C is a number of classes.
    This function splits input mask into C masks. Each mask contains separate instances of this class
    labeled starting from 1 and 0 as background.

    Example:
        Input mask (C = 2):
            [[0, 0, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0]]

        Output:
            [[[1, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [2, 2, 0, 1]],

            [[0, 0, 1, 1],
            [2, 0, 0, 0],
            [2, 2, 2, 0],
            [0, 0, 2, 0]]]

    Args:
        mask (numpy.ndarray): Mask of shape (H x W). Each cell contains contains cell's class number.

    Returns:
        numpy.ndarray: Labeled mask of shape (C x H x W).

    """
    labeled_channels = []
    for label_nr in range(0, mask.max() + 1):
        labeled_channels.append(label(mask == label_nr))
    labeled_image = np.stack(labeled_channels)
    return labeled_image


def label_multilayer_image(mask):
    labeled_channels = []
    for channel in mask:
        labeled_channels.append(label(channel))
    labeled_image = np.stack(labeled_channels)
    return labeled_image


def erode_image(mask, erode_selem_size):
    """Erode mask.

    Args:
        mask (numpy.ndarray): Mask of shape (H x W) or multiple masks of shape (C x H x W).
        erode_selem_size (int): Size of rectangle structuring element used for erosion.

    Returns:
        numpy.ndarray: Eroded mask of shape (H x W) or multiple masks of shape (C x H x W).

    """
    if not erode_selem_size > 0:
        return mask
    selem = rectangle(erode_selem_size, erode_selem_size)
    if mask.ndim == 2:
        eroded_image = erosion(mask, selem=selem)
    else:
        eroded_image = []
        for category_mask in mask:
            eroded_image.append(erosion(category_mask, selem=selem))
            eroded_image = np.stack(eroded_image)
    return add_dropped_objects(mask, eroded_image)


def dilate_image(mask, dilate_selem_size):
    """Dilate mask.

    Args:
        mask (numpy.ndarray): Mask of shape (H x W) or multiple masks of shape (C x H x W).
        dilate_selem_size (int): Size of rectangle structuring element used for dilation.

    Returns:
        numpy.ndarray: dilated Mask of shape (H x W) or multiple masks of shape (C x H x W).

    """
    if not dilate_selem_size > 0:
        return mask
    selem = rectangle(dilate_selem_size, dilate_selem_size)
    if mask.ndim == 2:
        dilated_image = dilation(mask, selem=selem)
    else:
        dilated_image = []
        for category_mask in mask:
            dilated_image.append(dilation(category_mask, selem=selem))
        dilated_image = np.stack(dilated_image)
    return dilated_image


def dense_crf(img, output_probs, compat_gaussian=3, sxy_gaussian=1,
              compat_bilateral=10, sxy_bilateral=1, srgb=50, iterations=5):
    """Perform fully connected CRF.

    This function performs CRF method described in the following paper:

        Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
        Philipp Krähenbühl and Vladlen Koltun
        NIPS 2011
        https://arxiv.org/abs/1210.5644

    Args:
        img (numpy.ndarray): RGB image of shape (3 x H x W).
        output_probs (numpy.ndarray): Probability map of shape (C x H x W).
        compat_gaussian: Compat value for Gaussian case.
        sxy_gaussian: x/y standard-deviation, theta_gamma from the CRF paper.
        compat_bilateral: Compat value for RGB case.
        sxy_bilateral: x/y standard-deviation, theta_alpha from the CRF paper.
        srgb: RGB standard-deviation, theta_beta from the CRF paper.
        iterations: Number of CRF iterations.

    Returns:
        numpy.ndarray: Probability map of shape (C x H x W) after applying CRF.

    """
    height = output_probs.shape[1]
    width = output_probs.shape[2]

    crf = DenseCRF2D(width, height, 2)
    unary = unary_from_softmax(output_probs)
    org_img = denormalize_img(img, mean=MEAN, std=STD) * 255.
    org_img = org_img.transpose(1, 2, 0)
    org_img = np.ascontiguousarray(org_img, dtype=np.uint8)

    crf.setUnaryEnergy(unary)

    crf.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
    crf.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb, rgbim=org_img, compat=compat_bilateral)

    crf_image = crf.inference(iterations)
    crf_image = np.array(crf_image).reshape(output_probs.shape)

    return crf_image


def build_score(image, probabilities):
    total_score = []
    for category_instances, category_probabilities in zip(image, probabilities):
        score = []
        for label_nr in range(1, category_instances.max() + 1):
            masked_instance = np.ma.masked_array(category_probabilities, mask=category_instances != label_nr)
            score.append(masked_instance.mean() * np.sqrt(np.count_nonzero(category_instances == label_nr)))
        total_score.append(score)
    return image, total_score


def crop_image_center_per_class(image, h_crop, w_crop):
    """Crop image center.

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        h_crop: Height of a cropped image.
        w_crop: Width of a cropped image.

    Returns:
        numpy.ndarray: Cropped image of shape (C x H x W).

    """
    cropped_per_class_prediction = []
    for class_prediction in image:
        h, w = class_prediction.shape[:2]
        h_start, w_start = int((h - h_crop) / 2.), int((w - w_crop) / 2.)
        cropped_prediction = class_prediction[h_start:-h_start, w_start:-w_start]
        cropped_per_class_prediction.append(cropped_prediction)
    cropped_per_class_prediction = np.stack(cropped_per_class_prediction)
    return cropped_per_class_prediction


def join_score_image(image, score):
    return (image, score)


def get_features_for_image(image, probabilities, annotations):
    image_features = []
    category_layers_inds = np.cumsum(CATEGORY_LAYERS)
    thresholds = get_thresholds()
    for category_ind, category_instances in enumerate(image):
        layer_features = []
        category_nr = np.searchsorted(category_layers_inds, category_ind, side='right')
        category_probabilities = probabilities[category_nr]
        threshold = round(thresholds[category_ind], 2)
        category_annotations = annotations.get(CATEGORY_IDS[category_nr], [])
        iou_matrix = get_iou_matrix(category_instances, category_annotations)
        for label_nr in range(1, category_instances.max() + 1):
            mask = category_instances == label_nr
            iou = get_iou(iou_matrix, label_nr)
            mask_probabilities = np.where(mask, category_probabilities, 0)
            area = np.count_nonzero(mask)
            mean_prob = mask_probabilities.sum() / area
            max_prob = mask_probabilities.max()
            bbox = get_bbox(mask)
            bbox_height = bbox[1] - bbox[0]
            bbox_width = bbox[3] - bbox[2]
            bbox_aspect_ratio = bbox_height / bbox_width
            bbox_area = bbox_width * bbox_height
            bbox_fill = area / bbox_area
            dist_to_border = get_distance_to_border(bbox, mask.shape)
            contour_length = get_contour_length(mask)
            mask_features = {'iou': iou, 'threshold': threshold, 'area': area, 'mean_prob': mean_prob,
                             'max_prob': max_prob, 'bbox_ar': bbox_aspect_ratio,
                             'bbox_area': bbox_area, 'bbox_fill': bbox_fill, 'dist_to_border': dist_to_border,
                             'contour_length': contour_length}
            layer_features.append(mask_features)
        image_features.append(pd.DataFrame(layer_features))
    return image_features


def get_iou_matrix(labels, annotations):
    mask_anns = []
    if annotations is None or annotations == []:
        return None
    else:
        for annotation in annotations:
            if not isinstance(annotation['segmentation'], dict):
                annotation['segmentation'] = \
                    cocomask.frPyObjects(annotation['segmentation'], labels.shape[0], labels.shape[1])[0]
        annotations = [annotation['segmentation'] for annotation in annotations]
        for label_nr in range(1, labels.max() + 1):
            mask = labels == label_nr
            mask_anns.append(rle_from_binary(mask.astype('uint8')))
        iou_matrix = cocomask.iou(mask_anns, annotations, [0, ] * len(annotations))
        return iou_matrix


def get_iou(iou_matrix, label_nr):
    if iou_matrix is not None:
        return iou_matrix[label_nr - 1].max()
    else:
        return None


def get_thresholds():
    thresholds = []
    for n_thresholds in CATEGORY_LAYERS:
        thrs_step = 1. / (n_thresholds + 1)
        category_thresholds = np.arange(thrs_step, 1, thrs_step)
        thresholds.extend(category_thresholds)
    return thresholds


def get_bbox(mask):
    '''taken from https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array and
    modified to prevent bbox of zero area'''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax + 1, cmin, cmax + 1


def get_distance_to_border(bbox, im_size):
    return min(bbox[0], im_size[0] - bbox[1], bbox[2], im_size[1] - bbox[3])


def get_contour(mask):
    mask_contour = np.zeros_like(mask).astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask_contour, contours, -1, (255, 255, 255), 1)
    return mask_contour


def get_contour_length(mask):
    return np.count_nonzero(get_contour(mask))


def remove_overlapping_masks(image, scores, iou_threshold=0.5):
    scores_with_labels = []
    for layer_nr, layer_scores in enumerate(scores):
        scores_with_labels.extend([(score, layer_nr, label_nr + 1) for label_nr, score in enumerate(layer_scores)])
    scores_with_labels.sort(key=lambda x: x[0], reverse=True)
    for i, (score_i, layer_nr_i, label_nr_i) in enumerate(scores_with_labels):
        base_mask = image[layer_nr_i] == label_nr_i
        for score_j, layer_nr_j, label_nr_j in scores_with_labels[i + 1:]:
            mask_to_check = image[layer_nr_j] == label_nr_j
            iou = get_iou_for_mask_pair(base_mask, mask_to_check)
            if iou > iou_threshold:
                scores_with_labels.remove((score_j, layer_nr_j, label_nr_j))
                scores[layer_nr_j][label_nr_j - 1] = 0
    return image, scores


def get_iou_for_mask_pair(mask1, mask2):
    intersection = np.count_nonzero(mask1 * mask2)
    union = np.count_nonzero(mask1 + mask2)
    return intersection / union
