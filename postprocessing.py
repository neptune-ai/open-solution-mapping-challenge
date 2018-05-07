import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize
from tqdm import tqdm

from steps.base import BaseTransformer
from utils import categorize_image


class MulticlassLabeler(BaseTransformer):
    def transform(self, images):
        labeled_images = []
        for i, image in enumerate(images):
            labeled_image = label_multiclass_image(image)
            labeled_images.append(labeled_image)
        return {'labeled_images': labeled_images}


class Resizer(BaseTransformer):
    def transform(self, images, target_sizes):
        resized_images = []
        for image, target_size in tqdm(zip(images, target_sizes)):
            n_channels = image.shape[0]
            resized_image = resize(image, (n_channels,) + target_size, mode='constant')
            resized_images.append(resized_image)
        return {'resized_images': resized_images}


class CategoryMapper(BaseTransformer):
    def transform(self, images):
        categorized_images = []
        for image in tqdm(images):
            categorized_images.append(categorize_image(image))
        return {'categorized_images': categorized_images}


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled


def label_multiclass_image(mask):
    labeled_channels = []
    for label_nr in range(0, mask.max() + 1):
        labeled_channels.append(label(mask == label_nr))
    labeled_image = np.stack(labeled_channels)
    return labeled_image
