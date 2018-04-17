import numpy as np
import skimage.morphology as morph
from scipy import ndimage as ndi
from scipy.stats import itemfreq
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.externals import joblib
from tqdm import tqdm

from steps.base import BaseTransformer
from utils import relabel


class Resizer(BaseTransformer):
    def transform(self, images, target_sizes):
        resized_images = []
        for image, target_size in tqdm(zip(images, target_sizes)):
            resized_image = resize(image, target_size, mode='constant')
            resized_images.append(resized_image)
        return {'resized_images': resized_images}


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


class Postprocessor(BaseTransformer):
    def __init__(self, **kwargs):
        pass

    def transform(self, images, contours):
        labeled_images = []
        for image, contour in tqdm(zip(images, contours)):
            labeled_image = postprocess(image, contour)
            labeled_images.append(labeled_image)
        return {'labeled_images': labeled_images}


class CellSizer(BaseTransformer):
    def __init__(self, **kwargs):
        pass

    def transform(self, labeled_images):
        mean_sizes = []
        for image in tqdm(labeled_images):
            mean_size = mean_cell_size(image)
            mean_sizes.append(mean_size)
        return {'sizes': mean_sizes}


def watershed_center(image, center):
    distance = ndi.distance_transform_edt(image)
    markers, nr_blobs = ndi.label(center)
    labeled = morph.watershed(-distance, markers, mask=image)

    dropped, _ = ndi.label(image - (labeled > 0))
    dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
    correct_labeled = dropped + labeled
    return relabel(correct_labeled)


def watershed_contour(image, contour):
    mask = np.where(contour == 1, 0, image)

    distance = ndi.distance_transform_edt(mask)
    markers, nr_blobs = ndi.label(mask)
    labeled = morph.watershed(-distance, markers, mask=image)

    dropped, _ = ndi.label(image - (labeled > 0))
    dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
    correct_labeled = dropped + labeled
    return relabel(correct_labeled)


def postprocess(image, contour):
    cleaned_mask = get_clean_mask(image, contour)
    good_markers = get_markers(cleaned_mask, contour)
    good_distance = get_distance(cleaned_mask)

    labels = morph.watershed(-good_distance, good_markers, mask=cleaned_mask)

    labels = add_dropped_water_blobs(labels, cleaned_mask)

    min_joinable_size = min_blob_size(image > 0.5, percentile=50, fraction_of_percentile=0.2)
    labels = connect_small(labels, min_cell_size=min_joinable_size)

    min_acceptable_size = min_blob_size(image > 0.5, percentile=50, fraction_of_percentile=0.1)
    labels = drop_small(labels, min_size=min_acceptable_size)

    labels = drop_big_artifacts(labels, scale=0.01)

    return relabel(labels)


def drop_artifacts_per_label(labels, initial_mask):
    labels_cleaned = np.zeros_like(labels)
    for i in range(1, labels.max() + 1):
        component = np.where(labels == i, 1, 0)
        component_initial_mask = np.where(labels == i, initial_mask, 0)
        component = drop_artifacts(component, component_initial_mask)
        labels_cleaned = labels_cleaned + component * i
    return labels_cleaned


def get_clean_mask(m, c):
    m_b = m > 0.5
    c_b = c > 0.5
    m_ = np.where(m_b | c_b, 1, 0)

    clean_mask = np.zeros_like(m)
    labels, label_nr = ndi.label(m_)
    for label in range(1, label_nr + 1):
        mask_component = np.where(labels == label, m_b, 0)
        contour_component = np.where(labels == label, c_b, 0)

        component_radius = np.sqrt(mask_component.sum())
        struct_size = int(max(0.05 * component_radius, 5))
        struct_el = morph.disk(struct_size)
        m_padded = pad_mask(mask_component, pad=struct_size)
        m_padded = morph.binary_closing(m_padded, selem=struct_el)
        m_padded = morph.binary_opening(m_padded, selem=struct_el)
        mask_component_ = crop_mask(m_padded, crop=struct_size)
        mask_component_ = ndi.binary_fill_holes(mask_component_)

        mask_component_ = np.where(mask_component_ | mask_component | contour_component, 1, 0)
        clean_mask += mask_component_

    clean_mask = np.where(clean_mask, 1, 0)
    return clean_mask


def get_markers(m_b, c):
    c_b = c > 0.75
    marker_component = np.where(m_b & ~c_b, 1, 0)
    min_size = min_blob_size(m_b, percentile=50, fraction_of_percentile=0.2)
    labels, label_nr = ndi.label(marker_component)
    markers = np.zeros_like(marker_component)
    for label in range(1, label_nr + 1):
        mask_component = np.where(labels == label, 1, 0)

        if mask_component.sum() < min_size:
            continue

        mask_component = ndi.binary_fill_holes(mask_component)

        if mask_component.sum() < min_size * 3:
            markers += np.where(mask_component, 1, 0)
            continue

        component_radius = np.sqrt(mask_component.sum())
        struct_size = int(component_radius * 0.15)
        struct_el = morph.disk(struct_size)
        m_padded = pad_mask(mask_component, pad=struct_size)
        m_padded = morph.binary_erosion(m_padded, selem=struct_el)
        mask_component = crop_mask(m_padded, crop=struct_size)

        mask_component_labels, _ = ndi.label(mask_component)
        mask_component = drop_small(mask_component_labels, min_size)
        markers += np.where(mask_component, 1, 0)

    markers, _ = ndi.label(markers)
    return markers


def get_distance(m_b):
    distance = ndi.distance_transform_edt(m_b)
    return distance


def add_dropped_water_blobs(water, mask_cleaned):
    water_mask = (water > 0).astype(np.uint8)
    dropped = mask_cleaned - water_mask
    dropped, _ = ndi.label(dropped)
    dropped = np.where(dropped, dropped + water.max(), 0)
    water = water + dropped
    return water


def fill_holes_per_blob(image):
    image_cleaned = np.zeros_like(image)
    for i in range(1, image.max() + 1):
        mask = np.where(image == i, 1, 0)
        mask = ndi.morphology.binary_fill_holes(mask)
        image_cleaned = image_cleaned + mask * i
    return image_cleaned


def drop_artifacts(mask_after, mask_pre, min_coverage=0.5):
    connected, nr_connected = ndi.label(mask_after)
    mask = np.zeros_like(mask_after)
    for i in range(1, nr_connected + 1):
        conn_blob = np.where(connected == i, 1, 0)
        initial_space = np.where(connected == i, mask_pre, 0)
        blob_size = np.sum(conn_blob)
        initial_blob_size = np.sum(initial_space)
        coverage = float(initial_blob_size) / float(blob_size)
        if coverage > min_coverage:
            mask = mask + conn_blob
        else:
            mask = mask + initial_space
    return mask


def mean_blob_size(mask):
    labels, labels_nr = ndi.label(mask)
    if labels_nr < 2:
        mean_area = 1
        mean_radius = 1
    else:
        blob_sizes = itemfreq(labels)
        blob_sizes = blob_sizes[blob_sizes[:, 0].argsort()][1:, :]
        mean_area = int(blob_sizes.mean())
        mean_radius = int(np.round(np.sqrt(mean_area) / np.pi))
    return mean_area, mean_radius


def pad_mask(mask, pad):
    if pad <= 1:
        pad = 2
    h, w = mask.shape
    h_pad = h + 2 * pad
    w_pad = w + 2 * pad
    mask_padded = np.zeros((h_pad, w_pad))
    mask_padded[pad:pad + h, pad:pad + w] = mask
    mask_padded[pad, :] = 1
    mask_padded[pad + h + 1, :] = 1
    mask_padded[:, pad] = 1
    mask_padded[:, pad + w + 1] = 1

    return mask_padded


def crop_mask(mask, crop):
    if crop <= 1:
        crop = 2
    h, w = mask.shape
    mask_cropped = mask[crop:h - crop, crop:w - crop]
    return mask_cropped


def drop_small(img, min_size):
    img = morph.remove_small_objects(img, min_size=min_size)
    return relabel(img)


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled


def min_blob_size(mask, percentile=25, fraction_of_percentile=0.1):
    labels, labels_nr = ndi.label(mask)
    if labels_nr < 2:
        return 0
    else:
        blob_sizes = itemfreq(labels)
        blob_sizes = blob_sizes[blob_sizes[:, 0].argsort()][1:, 1]
        return fraction_of_percentile * np.percentile(blob_sizes, percentile)


def mean_cell_size(labeled_image):
    blob_sizes = itemfreq(labeled_image)
    if blob_sizes.shape[0] == 1:
        return 0
    else:
        blob_sizes = blob_sizes[blob_sizes[:, 0].argsort()][1:, 1]
        return np.mean(blob_sizes)


def find_touching_labels(labels, label_id):
    mask = np.where(labels == label_id, 0, 1)
    dist = ndi.distance_transform_edt(mask)
    neighbour_labels = np.unique(np.where(dist == 1.0, labels, 0)).tolist()
    neighbour_labels.remove(0)

    neighbour_labels_with_sizes = [(neighbor_label, np.where(labels == neighbor_label, 1, 0).sum())
                                   for neighbor_label in neighbour_labels]
    neighbour_labels_with_sizes = sorted(neighbour_labels_with_sizes,
                                         key=lambda x: x[1],
                                         reverse=False)
    neighbour_labels_sorted = [neighbor_label for neighbor_label, _ in neighbour_labels_with_sizes]
    neighbour_labels_sorted
    return neighbour_labels_sorted


def connect_small(labels, min_cell_size=None):
    labels_with_sizes = [(label_id, np.where(labels == label_id, 1, 0).sum())
                         for label_id in range(1, labels.max() + 1)]
    label_ids_sorted_by_size = [lws[0] for lws in sorted(labels_with_sizes,
                                                         key=lambda x: x[1],
                                                         reverse=False)]
    touching_cell_was_connected = False
    for label_id in label_ids_sorted_by_size:
        cell_size = np.sum(labels == label_id)
        touching_labels = find_touching_labels(labels, label_id)
        for touching_label in touching_labels:
            touching_cell_mask = np.where(labels == touching_label, 1, 0)
            touching_cell_size = np.sum(touching_cell_mask)
            if touching_cell_size < min_cell_size:
                labels = np.where(labels == touching_label, label_id, labels)
                touching_cell_was_connected = True
    labels = relabel(labels)
    if touching_cell_was_connected:
        labels = connect_small(labels, min_cell_size)
    return relabel(labels)


def is_slim(im, object_ar, area_ar):
    ind = np.where(im == 1)
    ydiff = np.max(ind[0]) - np.min(ind[0])
    xdiff = np.max(ind[1]) - np.min(ind[1])
    rec_area = xdiff * ydiff
    area = np.sum(im == 1)
    if xdiff / ydiff < object_ar and xdiff / ydiff > 1.0 / object_ar and area / rec_area > area_ar:
        return False
    return True


def touching_edges(im, margin):
    indices = np.where(im == 1)
    edges = []
    edges.append(np.sum(indices[0] <= margin))
    edges.append(np.sum(indices[1] <= margin))
    edges.append(np.sum(indices[0] >= im.shape[0] - 1 - margin))
    edges.append(np.sum(indices[1] >= im.shape[1] - 1 - margin))
    return np.sum(np.array(edges) > 0)


def drop_big_artifacts(im, scale):
    im_cleaned = np.copy(im)
    im_size = im.shape[0] * im.shape[1]
    for label in np.unique(im):
        if label == 0:
            continue
        size = np.sum(im == label)
        if size < scale * im_size:
            continue
        if not is_slim(im == label, 2, 0.5):
            continue
        if touching_edges(im=im == label, margin=2) < 2:
            continue
        im_cleaned[im_cleaned == label] = 0
    return im_cleaned
