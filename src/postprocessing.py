import numpy as np
from skimage.transform import resize
from skimage.morphology import erosion, dilation, rectangle
from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax

from .utils import denormalize_img, add_dropped_objects, label
from .pipeline_config import MEAN, STD


def resize_image(image, target_size):
    """Resize image to target size

    Args:
        image (numpy.ndarray): image of shape (C x H x W)
        target_size (tuple): target size (H, W)

    Returns:
        numpy.ndarray: resized image of shape (C x H x W)

    """
    n_channels = image.shape[0]
    resized_image = resize(image, (n_channels,) + target_size, mode='constant')
    return resized_image


def categorize_image(image):
    """Maps probability map to categories. Each pixel is assigned with a category with highest probability.

    Args:
        image (numpy.ndarray): probability map of shape (C x H x W)

    Returns:
        numpy.ndarray: categorized image of shape (H x W)

    """
    return np.argmax(image, axis=0)


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
        mask (numpy.ndarray): mask of shape (H x W). Each cell contains contains cell's class number.

    Returns:
        numpy.ndarray: labeled mask of shape (C x H x W)

    """
    labeled_channels = []
    for label_nr in range(0, mask.max() + 1):
        labeled_channels.append(label(mask == label_nr))
    labeled_image = np.stack(labeled_channels)
    return labeled_image


def erode_image(mask, erode_selem_size):
    """Erode mask.

    Args:
        mask (numpy.ndarray): mask of shape (H x W) or set of masks of shape (C x H x W)
        erode_selem_size (int): size of rectangle structuring element used for erosion

    Returns:
        numpy.ndarray: eroded mask of shape (H x W) or set of masks of shape (C x H x W)

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
        mask (numpy.ndarray): mask of shape (H x W) or set of masks of shape (C x H x W)
        dilate_selem_size (int): size of rectangle structuring element used for dilation

    Returns:
        numpy.ndarray: dilated mask of shape (H x W) or set of masks of shape (C x H x W)

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
        img (numpy.ndarray): RGB image of shape (3 x H x W)
        output_probs (numpy.ndarray): probability map of shape (C x H x W).
        compat_gaussian: compat value for Gaussian case
        sxy_gaussian: x/y standard-deviation, theta_gamma from the paper
        compat_bilateral: compat value for RGB case
        sxy_bilateral: x/y standard-deviation, theta_alpha from the paper
        srgb: rgb standard-deviation, theta_beta from the paper
        iterations: number of iterations

    Returns:
        numpy.ndarray: probability map of shape (C x H x W) after applying CRF

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
        image (numpy.ndarray): image of shape (C x H x W)
        h_crop: height of a cropped image
        w_crop: width of a cropped image

    Returns:
        numpy.ndarray: cropped image of shape (C x H x W)

    """
    cropped_per_class_prediction = []
    for class_prediction in image:
        h, w = class_prediction.shape[:2]
        h_start, w_start = int((h - h_crop) / 2.), int((w - w_crop) / 2.)
        cropped_prediction = class_prediction[h_start:-h_start, w_start:-w_start]
        cropped_per_class_prediction.append(cropped_prediction)
    cropped_per_class_prediction = np.stack(cropped_per_class_prediction)
    return cropped_per_class_prediction
