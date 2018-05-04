import os

import numpy as np
import torch
from imageio import imwrite
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from skimage.transform import resize
from tqdm import tqdm
from skimage.morphology import binary_erosion, rectangle

from utils import get_logger
from postprocessing import label

logger = get_logger()


def overlay_masks(data_dir, dataset, target_dir, category_ids, erode=None, is_small=False):
    if is_small:
        suffix = "-small"
    else:
        suffix = ""
    annotation_file_name = "annotation{}.json".format(suffix)
    annotation_file_path = os.path.join(data_dir, dataset, annotation_file_name)
    coco = COCO(annotation_file_path)
    image_ids = coco.getImgIds()
    for image_id in tqdm(image_ids):
        image = coco.loadImgs(image_id)[0]
        image_size = (image["height"], image["width"])
        mask_overlayed = np.zeros(image_size).astype('uint8')
        for category_nr, category_id in enumerate(category_ids):
            if category_id != None:
                annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=[category_id, ])
                annotations = coco.loadAnns(annotation_ids)
                mask = overlay_masks_from_annotations(annotations, image_size)
                if erode is not None:
                    mask_e = overlay_eroded_masks_from_annotations(annotations, image_size, erode)
                    mask = add_dropped_objects(mask, mask_e)
                mask_overlayed = np.where(mask, category_nr, mask_overlayed)
        target_filepath = os.path.join(target_dir, dataset, "masks", image["file_name"][:-4]) + ".png"
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        try:
            imwrite(target_filepath, mask_overlayed)
        except:
            logger.info("Failed to save image: {}".format(image_id))


def overlay_masks_from_annotations(annotations, image_size):
    mask = np.zeros(image_size)
    for ann in annotations:
        rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
        m = cocomask.decode(rle)
        m = m.reshape(image_size)
        mask += m
    return np.where(mask > 0, 1, 0).astype('uint8')


def overlay_eroded_masks_from_annotations(annotations, image_size, selem_size):
    mask = np.zeros(image_size)
    selem = rectangle(selem_size, selem_size)
    for ann in annotations:
        rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
        m = cocomask.decode(rle)
        m = m.reshape(image_size)
        m = binary_erosion(m, selem=selem)
        mask += m
    return np.where(mask > 0, 1, 0).astype('uint8')


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


def add_dropped_objects(original, processed):
    reconstructed = processed
    labeled = label(original)
    for i in range(1, labeled.max() + 1):
        if np.sum((original == i) * processed) == 0:
            reconstructed += (labeled == i).astype('uint8')
    return reconstructed
