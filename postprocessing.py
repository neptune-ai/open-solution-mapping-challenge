import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize
from skimage.morphology import binary_dilation, rectangle
from tqdm import tqdm

from steps.base import BaseTransformer
from utils import categorize_image, dense_crf


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


class MaskDilator(BaseTransformer):
    def __init__(self, dilate_selem_size):
        self.selem_size = dilate_selem_size

    def transform(self, images):
        dilated_images = []
        for image in tqdm(images):
            dilated_images.append(dilate_image(image, self.selem_size))
        return {'categorized_images': dilated_images}


class DenseCRF(BaseTransformer):
    def __init__(self, compat_gaussian, sxy_gaussian, compat_bilateral, sxy_bilateral, srgb, **kwargs):
        self.compat_gaussian = compat_gaussian
        self.sxy_gaussian = sxy_gaussian
        self.compat_bilateral = compat_bilateral
        self.sxy_bilateral = sxy_bilateral
        self.srgb = srgb

    def transform(self, images, org_images_gen):
        crf_images = []
        org_images = self.get_org_images(org_images_gen)
        for image, org_image in tqdm(zip(images, org_images)):
            crf_image = dense_crf(org_image, image, self.compat_gaussian, self.sxy_gaussian,
                                  self.compat_bilateral, self.sxy_bilateral, self.srgb)
            crf_images.append(crf_image)
        return {'crf_images': crf_images}

    def get_org_images(self, org_images_gen):
        org_images = []
        batch_gen, steps = org_images_gen
        for batch_id, data in enumerate(batch_gen):
            if isinstance(data, list):
                X = data[0]
            else:
                X = data
            for image in X.numpy():
                org_images.append(image)
            if batch_id == steps:
                break
        return org_images


class MulticlassLabelerStream(BaseTransformer):
    def transform(self, images):
        return {'labeled_images': self._transform(images)}

    def _transform(self, images):
        for i, image in enumerate(images):
            labeled_image = label_multiclass_image(image)
            yield labeled_image


class ResizerStream(BaseTransformer):
    def transform(self, images, target_sizes):
        return {'resized_images': self._transform(images, target_sizes)}

    def _transform(self, images, target_sizes):
        for image, target_size in tqdm(zip(images, target_sizes)):
            n_channels = image.shape[0]
            resized_image = resize(image, (n_channels,) + target_size, mode='constant')
            yield resized_image


class CategoryMapperStream(BaseTransformer):
    def transform(self, images):
        return {'categorized_images': self._transform(images)}

    def _transform(self, images):
        for image in tqdm(images):
            yield categorize_image(image)


class MaskDilatorStream(BaseTransformer):
    def __init__(self, dilate_selem_size):
        self.selem_size = dilate_selem_size

    def transform(self, images):
        return {'categorized_images': self._transform(images)}

    def _transform(self, images):
        for image in tqdm(images):
            yield dilate_image(image, self.selem_size)


class DenseCRFStream(BaseTransformer):
    def __init__(self, compat_gaussian, sxy_gaussian, compat_bilateral, sxy_bilateral, srgb, **kwargs):
        self.compat_gaussian = compat_gaussian
        self.sxy_gaussian = sxy_gaussian
        self.compat_bilateral = compat_bilateral
        self.sxy_bilateral = sxy_bilateral
        self.srgb = srgb

    def transform(self, images, org_images_gen):
        return {'crf_images': self._transform(images, org_images_gen)}

    def _transform(self, images, org_images_gen):
        org_images = self.get_org_images(org_images_gen)
        for image, org_image in tqdm(zip(images, org_images)):
            crf_image = dense_crf(org_image, image, self.compat_gaussian, self.sxy_gaussian,
                                  self.compat_bilateral, self.sxy_bilateral, self.srgb)
            yield crf_image

    def get_org_images(self, datagen):
        batch_gen, steps = datagen
        for batch_id, data in enumerate(batch_gen):
            if isinstance(data, list):
                X = data[0]
            else:
                X = data
            for image in X.numpy():
                yield image
            if batch_id == steps:
                break


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled


def label_multiclass_image(mask):
    labeled_channels = []
    for label_nr in range(0, mask.max() + 1):
        labeled_channels.append(label(mask == label_nr))
    labeled_image = np.stack(labeled_channels)
    return labeled_image


def dilate_image(mask, selem_size):
    selem = rectangle(selem_size, selem_size)
    return binary_dilation(mask, selem=selem)
