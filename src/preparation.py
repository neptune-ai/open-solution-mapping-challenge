from functools import partial
import multiprocessing as mp
import os

import numpy as np
import torch
from imageio import imwrite
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from skimage.transform import resize
from tqdm import tqdm
from skimage.morphology import binary_erosion, rectangle, binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
from sklearn.externals import joblib

from .utils import get_logger, add_dropped_objects, label

logger = get_logger()


def overlay_masks(data_dir, dataset, target_dir, category_ids, erode=0, dilate=0, is_small=False, nthreads=1,
                  border_width=0, small_annotations_size=14):
    if is_small:
        suffix = "-small"
    else:
        suffix = ""
    annotation_file_name = "annotation{}.json".format(suffix)
    annotation_file_path = os.path.join(data_dir, dataset, annotation_file_name)
    coco = COCO(annotation_file_path)
    image_ids = coco.getImgIds()

    _overlay_mask_one_image = partial(overlay_mask_one_image,
                                      dataset=dataset,
                                      target_dir=target_dir,
                                      coco=coco,
                                      category_ids=category_ids,
                                      erode=erode,
                                      dilate=dilate,
                                      border_width=border_width,
                                      small_annotations_size=small_annotations_size)

    process_nr = min(nthreads, len(image_ids))
    with mp.pool.ThreadPool(process_nr) as executor:
        executor.map(_overlay_mask_one_image, image_ids)


def overlay_mask_one_image(image_id, dataset, target_dir, coco, category_ids, erode, dilate, border_width,
                           small_annotations_size):
    image = coco.loadImgs(image_id)[0]
    image_size = (image["height"], image["width"])
    mask_overlayed = np.zeros(image_size).astype('uint8')
    distances = np.zeros(image_size)
    for category_nr, category_id in enumerate(category_ids):
        if category_id is not None:
            annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=[category_id, ])
            annotations = coco.loadAnns(annotation_ids)

            if erode < 0 or dilate < 0:
                raise ValueError('erode and dilate cannot be negative')

            if erode == 0:
                mask, distances = overlay_masks_from_annotations(annotations=annotations,
                                                                 image_size=image_size,
                                                                 distances=distances)
            elif dilate == 0:
                mask, _ = overlay_masks_from_annotations(annotations=annotations,
                                                         image_size=image_size)
                mask_eroded, distances = overlay_eroded_masks_from_annotations(annotations=annotations,
                                                                               image_size=image_size,
                                                                               erode=erode,
                                                                               distances=distances,
                                                                               small_annotations_size=small_annotations_size)
                mask = add_dropped_objects(mask, mask_eroded)
            else:
                mask, distances = overlay_eroded__dilated_masks_from_annotations(annotations=annotations,
                                                                                 image_size=image_size,
                                                                                 erode=erode,
                                                                                 dilate=dilate,
                                                                                 distances=distances,
                                                                                 small_annotations_size=small_annotations_size)
            mask_overlayed = np.where(mask, category_nr, mask_overlayed)

    sizes = get_size_matrix(mask_overlayed)
    distances, second_nearest_distances = clean_distances(distances)

    if border_width > 0:
        borders = (second_nearest_distances < border_width) & (~mask_overlayed)
        borders_class_id = mask_overlayed.max() + 1
        mask_overlayed = np.where(borders, borders_class_id, mask_overlayed)

    target_filepath = os.path.join(target_dir, dataset, "masks", os.path.splitext(image["file_name"])[0]) + ".png"
    target_filepath_dist = os.path.join(target_dir, dataset, "distances", os.path.splitext(image["file_name"])[0])
    target_filepath_sizes = os.path.join(target_dir, dataset, "sizes", os.path.splitext(image["file_name"])[0])
    os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(target_filepath_dist), exist_ok=True)
    os.makedirs(os.path.dirname(target_filepath_sizes), exist_ok=True)
    try:
        imwrite(target_filepath, mask_overlayed)
        joblib.dump(distances, target_filepath_dist)
        joblib.dump(sizes, target_filepath_sizes)
    except:
        logger.info("Failed to save image: {}".format(image_id))



def overlay_masks_from_annotations(annotations, image_size, distances=None):
    mask = np.zeros(image_size)
    for ann in annotations:
        rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
        m = cocomask.decode(rle)
        m = m.reshape(image_size)
        if is_on_border(m, 2):
            continue
        if distances is not None:
            distances = update_distances(distances, m)
        mask += m
    return np.where(mask > 0, 1, 0).astype('uint8'), distances


def overlay_eroded_masks_from_annotations(annotations, image_size, erode, distances, small_annotations_size):
    mask = np.zeros(image_size)
    for ann in annotations:
        rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
        m = cocomask.decode(rle)
        m = m.reshape(image_size)
        if is_on_border(m, 2):
            continue
        m_eroded = get_simple_eroded_mask(m, erode, small_annotations_size)
        if distances is not None:
            distances = update_distances(distances, m_eroded)
        mask += m_eroded
    return np.where(mask > 0, 1, 0).astype('uint8'), distances


def overlay_eroded__dilated_masks_from_annotations(annotations, image_size, erode, dilate, distances,
                                                   small_annotations_size):
    mask = np.zeros(image_size)
    for ann in annotations:
        rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
        m = cocomask.decode(rle)
        m = m.reshape(image_size)
        if is_on_border(m, 2):
            continue
        m_ = get_simple_eroded_dilated_mask(m, erode, dilate, small_annotations_size)
        if distances is not None:
            distances = update_distances(distances, m_)
        mask += m_
    return np.where(mask > 0, 1, 0).astype('uint8'), distances


def update_distances(dist, mask):
    if dist.sum() == 0:
        distances = distance_transform_edt(1 - mask)
    else:
        distances = np.dstack([dist, distance_transform_edt(1 - mask)])
    return distances


def clean_distances(distances):
    if len(distances.shape) < 3:
        distances = np.dstack([distances, distances])
    else:
        distances.sort(axis=2)
        distances = distances[:, :, :2]
    second_nearest_distances = distances[:, :, 1]
    distances_clean = np.sum(distances, axis=2)
    return distances_clean.astype(np.float16), second_nearest_distances


def preprocess_image(img, target_size=(128, 128)):
    img = resize(img, target_size, mode='constant')
    x = np.expand_dims(img, axis=0)
    x = x.transpose(0, 3, 1, 2)
    x = torch.FloatTensor(x)
    if torch.cuda.is_available():
        x = torch.autograd.Variable(x, volatile=True).cuda()
    else:
        x = torch.autograd.Variable(x, volatile=True)
    return x


def get_selem_size(mask, percent):
    mask_area = np.sum(mask)
    radius = np.sqrt(mask_area)
    result = max(2, int(radius * percent / 200))
    return result


def get_eroded_mask(mask, percent):
    if np.sum(mask) == 0:
        return mask
    diff = 100
    new_percent = percent
    iterations = 0
    while abs(diff) > 5 and iterations < 4:
        selem_size = get_selem_size(mask, new_percent)
        selem = rectangle(selem_size, selem_size)
        mask_eroded = binary_erosion(mask, selem=selem)
        mask_area = np.sum(mask)
        mask_eroded_area = np.sum(mask_eroded)
        percent_obtained = 100 * (1 - mask_eroded_area / mask_area)
        diff = percent - percent_obtained
        new_percent += diff
        iterations += 1
    if iterations > 3 and abs(diff) > 5:
        if diff < 0 and selem_size > 2:
            selem_size -= 1
        elif diff > 0:
            selem_size += 1
        selem = rectangle(selem_size, selem_size)
        mask_eroded = binary_erosion(mask, selem)
    return mask_eroded


def get_simple_eroded_mask(mask, selem_size, small_annotations_size):
    if mask.sum() > small_annotations_size**2:
        selem = rectangle(selem_size, selem_size)
        mask_eroded = binary_erosion(mask, selem=selem)
    else:
        mask_eroded = mask
    return mask_eroded


def get_simple_eroded_dilated_mask(mask, erode_selem_size, dilate_selem_size, small_annotations_size):
    if mask.sum() > small_annotations_size**2:
        selem = rectangle(erode_selem_size, erode_selem_size)
        mask_ = binary_erosion(mask, selem=selem)
    else:
        selem = rectangle(dilate_selem_size, dilate_selem_size)
        mask_ = binary_dilation(mask, selem=selem)
    return mask_


def get_size_matrix(mask):
    sizes = np.ones_like(mask)
    labeled = label(mask)
    for label_nr in range(1, labeled.max() + 1):
        label_size = (labeled == label_nr).sum()
        sizes = np.where(labeled == label_nr, label_size, sizes)
    return sizes

def is_on_border(mask, border_width):
    return not np.any(mask[border_width:-border_width, border_width:-border_width])