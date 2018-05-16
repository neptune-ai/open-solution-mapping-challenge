import numpy as np
from skimage.transform import resize
from skimage.morphology import binary_dilation, binary_erosion, dilation, rectangle
from tqdm import tqdm
from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax

from steps.base import BaseTransformer
from utils import categorize_image, denormalize_img, add_dropped_objects, label
from pipeline_config import MEAN, STD


class MulticlassLabeler(BaseTransformer):
    def transform(self, images):
        labeled_images = []
        for i, image in tqdm(enumerate(images)):
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


class MaskEroder(BaseTransformer):
    def __init__(self, erode_selem_size, **kwargs):
        self.selem_size = erode_selem_size

    def transform(self, images):
        if self.selem_size > 0:
            eroded_images = []
            for image in tqdm(images):
                eroded_images.append(erode_image(image, self.selem_size))
        else:
            eroded_images = images
        return {'eroded_images': eroded_images}


class MaskDilator(BaseTransformer):
    def __init__(self, dilate_selem_size, **kwargs):
        self.selem_size = dilate_selem_size

    def transform(self, images):
        dilated_images = []
        for image in tqdm(images):
            dilated_images.append(dilate_image(image, self.selem_size))
        return {'categorized_images': dilated_images}


class LabeledMaskDilator(BaseTransformer):
    def __init__(self, dilate_selem_size, **kwargs):
        self.selem_size = dilate_selem_size

    def transform(self, images):
        if self.selem_size > 0:
            dilated_images = []
            for image in tqdm(images):
                dilated_images.append(dilate_labeled_image(image, self.selem_size))
        else:
            dilated_images = images
        return {'dilated_images': dilated_images}


class DenseCRF(BaseTransformer):
    def __init__(self, compat_gaussian, sxy_gaussian, compat_bilateral, sxy_bilateral, srgb, **kwargs):
        self.compat_gaussian = compat_gaussian
        self.sxy_gaussian = sxy_gaussian
        self.compat_bilateral = compat_bilateral
        self.sxy_bilateral = sxy_bilateral
        self.srgb = srgb

    def transform(self, images, raw_images_generator):
        crf_images = []
        original_images = self.get_original_images(raw_images_generator)
        for image, org_image in tqdm(zip(images, original_images)):
            crf_image = _dense_crf(org_image, image, self.compat_gaussian, self.sxy_gaussian,
                                  self.compat_bilateral, self.sxy_bilateral, self.srgb)
            crf_images.append(crf_image)
        return {'crf_images': crf_images}

    def get_original_images(self, datagen):
        original_images = []
        batch_gen, steps = datagen
        for batch_id, data in enumerate(batch_gen):
            if isinstance(data, list):
                X = data[0]
            else:
                X = data
            for image in X.numpy():
                original_images.append(image)
            if batch_id == steps:
                break
        return original_images


class ScoreBuilder(BaseTransformer):
    def transform(self, images, probabilities):
        scores = []
        for image, image_probabilities in tqdm(zip(images, probabilities)):
            scores.append(build_score(image, image_probabilities))
        return {'images': images,
                'scores': scores}


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

    def transform(self, images, raw_images_generator):
        return {'crf_images': self._transform(images, raw_images_generator)}

    def _transform(self, images, raw_images_generator):
        original_images = self.get_original_images(raw_images_generator)
        for image, org_image in tqdm(zip(images, original_images)):
            crf_image = _dense_crf(org_image, image, self.compat_gaussian, self.sxy_gaussian,
                                  self.compat_bilateral, self.sxy_bilateral, self.srgb)
            yield crf_image

    def get_original_images(self, datagen):
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


def label_multiclass_image(mask):
    labeled_channels = []
    for label_nr in range(0, mask.max() + 1):
        labeled_channels.append(label(mask == label_nr))
    labeled_image = np.stack(labeled_channels)
    return labeled_image


def erode_image(mask, selem_size):
    selem = rectangle(selem_size, selem_size)
    eroded_image = binary_erosion(mask, selem=selem)
    return add_dropped_objects(mask, eroded_image)


def dilate_image(mask, selem_size):
    selem = rectangle(selem_size, selem_size)
    return binary_dilation(mask, selem=selem)


def dilate_labeled_image(mask, selem_size):
    selem = rectangle(selem_size, selem_size)
    dilated_image = []
    for category_mask in mask:
        dilated_image.append(dilation(category_mask, selem=selem))
    return np.stack(dilated_image)


def _dense_crf(img, output_probs, compat_gaussian=3, sxy_gaussian=1, compat_bilateral=10, sxy_bilateral=1, srgb=50):
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

    crf_image = crf.inference(5)
    crf_image = np.array(crf_image).reshape(output_probs.shape)

    return crf_image


def build_score(image, probabilities):
    total_score = []
    for category_instances, category_probabilities in zip(image, probabilities):
        score = []
        for label_nr in range(1, category_instances.max() + 1):
            masked_instance = np.ma.masked_array(category_probabilities, mask=category_instances != label_nr)
            score.append(masked_instance.mean()*np.sqrt(np.count_nonzero(category_instances == label_nr)))
        total_score.append(score)
    return total_score
