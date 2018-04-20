import numpy as np
from scipy import ndimage as ndi

from steps.base import BaseTransformer


class Thresholder(BaseTransformer):
    def __init__(self, threshold):
        self.threshold = threshold

    def transform(self, images):
        binarized_images = []
        for image in images:
            binarized_image = (image > self.threshold).astype(np.uint8)
            binarized_images.append(binarized_image)
        return {'binarized_images': binarized_images}


class BuildingLabeler(BaseTransformer):
    def transform(self, images):
        labeled_images = []
        for i, image in enumerate(images):
            labeled_image = label(image)
            labeled_images.append(labeled_image)

        return {'labeled_images': labeled_images}


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled
