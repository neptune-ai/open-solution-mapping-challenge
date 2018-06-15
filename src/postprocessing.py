import numpy as np
from skimage.transform import resize
from skimage.morphology import binary_dilation, binary_erosion, dilation, rectangle
from tqdm import tqdm
from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax
from pycocotools import mask as cocomask
import pandas as pd
import cv2

from steps.base import BaseTransformer
from utils import denormalize_img, add_dropped_objects, label, rle_from_binary, generate_data_frame_chunks
from pipeline_config import MEAN, STD, CATEGORY_LAYERS, CATEGORY_IDS


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
            crf_image = dense_crf(org_image, image, self.compat_gaussian, self.sxy_gaussian,
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


class PredictionCrop(BaseTransformer):
    def __init__(self, h_crop, w_crop):
        self.h_crop = h_crop
        self.w_crop = w_crop

    def transform(self, images):
        cropped_per_class_predictions = []
        for image in tqdm(images):
            cropped_per_class_prediction = crop_image_center_per_class(image, (self.h_crop, self.w_crop))
            cropped_per_class_predictions.append(cropped_per_class_prediction)
        return {'cropped_images': cropped_per_class_predictions}


class ScoreBuilder(BaseTransformer):
    def transform(self, images, probabilities):
        images_with_scores = []
        for image, image_probabilities in tqdm(zip(images, probabilities)):
            images_with_scores.append((image, build_score(image, image_probabilities)))
        return {'images_with_scores': images_with_scores}


class FeatureExtractor(BaseTransformer):
    def transform(self, images, probabilities, annotations=None):
        all_features = []
        if annotations is None:
            annotations = [{}] * len(images)
        for image, image_probabilities, image_annotations in tqdm(zip(images, probabilities, annotations)):
            all_features.append(get_features_for_image(image, image_probabilities, image_annotations))
        return {'features': all_features}


class ScoreImageJoiner(BaseTransformer):
    def transform(self, images, scores):
        images_with_scores = []
        for image, score in tqdm(zip(images, scores)):
            images_with_scores.append((image, score))
        return {'images_with_scores': images_with_scores}


class NonMaximumSupression(BaseTransformer):
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def transform(self, images_with_scores):
        cleaned_images_with_scores = []
        for image, scores in images_with_scores:
            cleaned_images_with_scores.append(remove_overlapping_masks(image, scores, self.iou_threshold))
        return {'images_with_scores': cleaned_images_with_scores}


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


class MaskEroderStream(BaseTransformer):
    def __init__(self, erode_selem_size, **kwargs):
        self.selem_size = erode_selem_size

    def transform(self, images):
        if self.selem_size > 0:
            return {'eroded_images': self._transform(images)}
        else:
            return {'eroded_images': images}

    def _transform(self, images):
        for image in tqdm(images):
            yield erode_image(image, self.selem_size)


class MaskDilatorStream(BaseTransformer):
    def __init__(self, dilate_selem_size):
        self.selem_size = dilate_selem_size

    def transform(self, images):
        return {'dilated_images': self._transform(images)}

    def _transform(self, images):
        for image in tqdm(images):
            yield dilate_image(image, self.selem_size)


class LabeledMaskDilatorStream(BaseTransformer):
    def __init__(self, dilate_selem_size, **kwargs):
        self.selem_size = dilate_selem_size

    def transform(self, images):
        if self.selem_size > 0:
            return {'dilated_images': self._transform(images)}
        else:
            return {'dilated_images': images}

    def _transform(self, images):
        for image in tqdm(images):
            yield dilate_labeled_image(image, self.selem_size)


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
            crf_image = dense_crf(org_image, image, self.compat_gaussian, self.sxy_gaussian,
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


class PredictionCropStream(BaseTransformer):
    def __init__(self, h_crop, w_crop):
        self.h_crop = h_crop
        self.w_crop = w_crop

    def transform(self, images):
        return {'cropped_images': self._transform(images)}

    def _transform(self, images):
        for image in tqdm(images):
            yield crop_image_center_per_class(image, (self.h_crop, self.w_crop))


class ScoreBuilderStream(BaseTransformer):
    def transform(self, images, probabilities):
        return {'images_with_scores': self._transform(images, probabilities)}

    def _transform(self, images, probabilities):
        for image, image_probabilities in tqdm(zip(images, probabilities)):
            yield (image, build_score(image, image_probabilities))


def label_multiclass_image(mask):
    labeled_channels = []
    for channel in mask:
        labeled_channels.append(label(channel))
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
    category_layers_inds = np.cumsum(CATEGORY_LAYERS)
    for category_ind, category_instances in enumerate(image):
        category_nr = np.searchsorted(category_layers_inds, category_ind, side='right')
        category_probabilities = probabilities[category_nr]
        score = []
        for label_nr in range(1, category_instances.max() + 1):
            mask = category_instances == label_nr
            area = np.count_nonzero(mask)
            mean_prob = np.where(mask, category_probabilities, 0).sum() / area
            score.append(mean_prob * np.sqrt(area))
        total_score.append(score)
    return total_score


def crop_image_center_per_class(image, size):
    h_crop, w_crop = size
    cropped_per_class_prediction = []
    for class_prediction in image:
        h, w = class_prediction.shape[:2]
        h_start, w_start = int((h - h_crop) / 2.), int((w - w_crop) / 2.)
        cropped_prediction = class_prediction[h_start:-h_start, w_start:-w_start]
        cropped_per_class_prediction.append(cropped_prediction)
    cropped_per_class_prediction = np.stack(cropped_per_class_prediction)
    return cropped_per_class_prediction


def categorize_image(image):
    categorized_image = []
    for category_id, category_output in enumerate(image):
        thrs_step = 1. / (CATEGORY_LAYERS[category_id] + 1)
        thresholds = np.arange(thrs_step, 1, thrs_step)
        for thrs in thresholds:
            categorized_image.append(category_output > thrs)
    return np.stack(categorized_image)


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
            dist_to_boarder = get_distance_to_boarder(bbox, mask.shape)
            contour_length = get_contour_length(mask)
            mask_features = {'iou': iou, 'threshold': threshold, 'area': area, 'mean_prob': mean_prob,
                             'max_prob': max_prob, 'label_nr': label_nr, 'bbox_ar': bbox_aspect_ratio,
                             'bbox_area': bbox_area, 'bbox_fill': bbox_fill, 'dist_to_boarder': dist_to_boarder,
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


def get_distance_to_boarder(bbox, im_size):
    return min(bbox[0], im_size[0] - bbox[1], bbox[2], im_size[1] - bbox[3])


def get_contour(mask):
    mask_contour = np.zeros_like(mask).astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask_contour, contours, -1, (255, 255, 255), 1)
    return mask_contour


def get_contour_length(mask):
    return np.count_nonzero(get_contour(mask))


def remove_overlapping_masks(image, scores, iou_threshold):
    scores_with_labels = []
    for layer_nr, layer_scores in enumerate(scores):
        scores_with_labels.extend([(score, layer_nr, label_nr + 1) for label_nr, score in enumerate(layer_scores)])
    scores_with_labels.sort(key=lambda x: x[0], reverse=True)
    for i, (score_i, layer_nr_i, label_nr_i) in enumerate(scores_with_labels):
        base_mask = image[label_nr_i] == label_nr_i
        for score_j, layer_nr_j, label_nr_j in scores_with_labels[i+1:]:
            mask_to_check = image[label_nr_j] == label_nr_j
            iou = get_iou_for_mask_pair(base_mask, mask_to_check)
            if iou > iou_threshold:
                scores_with_labels.remove((score_j, layer_nr_j, label_nr_j))
                scores[label_nr_j][label_nr_j - 1] = 0
    return image, scores


def get_iou_for_mask_pair(mask1, mask2):
    intersection = np.count_nonzero(mask1 * mask2)
    union = np.count_nonzero(mask1 + mask2)
    return intersection/union