import numpy as np
from skimage.transform import resize
from skimage.morphology import binary_dilation, binary_erosion, dilation, rectangle
from tqdm import tqdm
from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax

from .steps.base import BaseTransformer
from .utils import categorize_image, denormalize_img, add_dropped_objects, label
from .pipeline_config import MEAN, STD


class Resizer(BaseTransformer):
    """Creates transformer that resize images to target sizes."""

    def transform(self, images, target_sizes):
        """Resize images to target sizes.

        Args:
            images (list): list of N images, each image is a numpy.ndarray of shape (C x H x W)
            target_sizes (list): list of N target sizes, each target size is a tuple (H, W)

        Returns:
            list: list of N resized images
        """
        resized_images = []
        for image, target_size in tqdm(zip(images, target_sizes)):
            n_channels = image.shape[0]
            resized_image = resize(image, (n_channels,) + target_size, mode='constant')
            resized_images.append(resized_image)
        return {'resized_images': resized_images}


class CategoryMapper(BaseTransformer):
    """Creates transformer that maps probability maps (e.g. output of a neural network) to categories."""
    def transform(self, images):
        """Maps probability maps to categories. Each pixel is assigned with a category with highest probability.

        Args:
            images (list): list of N probability maps.
                Probability map is a numpy.ndarray of shape (C x H x W), where C is a number of categories.
                Cell [i, h, w] contains a probability of a pixel (h, w) of an image belonging to class C_i.

        Returns:
            list: list of N categorized images, each is a numpy.ndarray of shape (H x W)

        """
        categorized_images = []
        for image in tqdm(images):
            categorized_images.append(categorize_image(image))
        return {'categorized_images': categorized_images}


def resize_image(image, target_size):
    n_channels = image.shape[0]
    resized_image = resize(image, (n_channels,) + target_size, mode='constant')
    return resized_image


def label_multiclass_image(mask):
    labeled_channels = []
    for label_nr in range(0, mask.max() + 1):
        labeled_channels.append(label(mask == label_nr))
    labeled_image = np.stack(labeled_channels)
    return labeled_image


def erode_image(mask, erode_selem_size):
    if not erode_selem_size > 0:
        return mask
    selem = rectangle(erode_selem_size, erode_selem_size)
    eroded_image = binary_erosion(mask, selem=selem)
    return add_dropped_objects(mask, eroded_image)


def dilate_image(mask, dilate_selem_size):
    if not dilate_selem_size > 0:
        return mask
    selem = rectangle(dilate_selem_size, dilate_selem_size)
    return binary_dilation(mask, selem=selem)


def dilate_labeled_image(mask, dilate_selem_size):
    if not dilate_selem_size > 0:
        return mask
    selem = rectangle(dilate_selem_size, dilate_selem_size)
    dilated_image = []
    for category_mask in mask:
        dilated_image.append(dilation(category_mask, selem=selem))
    return np.stack(dilated_image)


def dense_crf(img, output_probs, compat_gaussian=3, sxy_gaussian=1, compat_bilateral=10, sxy_bilateral=1, srgb=50):
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
            score.append(masked_instance.mean() * np.sqrt(np.count_nonzero(category_instances == label_nr)))
        total_score.append(score)
    return image, total_score


def crop_image_center_per_class(image, h_crop, w_crop):
    cropped_per_class_prediction = []
    for class_prediction in image:
        h, w = class_prediction.shape[:2]
        h_start, w_start = int((h - h_crop) / 2.), int((w - w_crop) / 2.)
        cropped_prediction = class_prediction[h_start:-h_start, w_start:-w_start]
        cropped_per_class_prediction.append(cropped_prediction)
    cropped_per_class_prediction = np.stack(cropped_per_class_prediction)
    return cropped_per_class_prediction
