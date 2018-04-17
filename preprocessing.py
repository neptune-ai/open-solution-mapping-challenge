import glob

import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from skimage.color import rgb2grey, rgb2hed
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from sklearn.externals import joblib
from tqdm import tqdm

from preparation import get_contour
from steps.base import BaseTransformer
from utils import from_pil, to_pil, clip


class ImageReader(BaseTransformer):
    def __init__(self, x_columns, y_columns):
        self.x_columns = x_columns
        self.y_columns = y_columns

    def transform(self, meta, train_mode, meta_valid=None):
        X, y = self._transform(meta, train_mode)
        if meta_valid is not None:
            X_valid, y_valid = self._transform(meta_valid, train_mode)
        else:
            X_valid, y_valid = None, None

        return {'X': X,
                'y': y,
                'X_valid': X_valid,
                'y_valid': y_valid}

    def _transform(self, meta, train_mode):
        X_ = meta[self.x_columns].values

        X = self.load_images(X_, grayscale=False)
        if train_mode:
            y_ = meta[self.y_columns].values
            y = self.load_images(y_, grayscale=True)
        else:
            y = None

        return X, y

    def load_images(self, image_filepaths, grayscale):
        X = []
        for i in range(image_filepaths.shape[1]):
            column = image_filepaths[:, i]
            X.append([])
            for img_filepath in tqdm(column):
                img = self.load_image(img_filepath, grayscale=grayscale)
                X[i].append(img)
        return X

    def load_image(self, img_filepath, grayscale):
        image = Image.open(img_filepath, 'r')
        if not grayscale:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        return image

    def load(self, filepath):
        params = joblib.load(filepath)
        self.columns_to_get = params['x_columns']
        self.target_columns = params['y_columns']
        return self

    def save(self, filepath):
        params = {'x_columns': self.x_columns,
                  'y_columns': self.y_columns
                  }
        joblib.dump(params, filepath)


class ImageReaderRescaler(BaseTransformer):
    def __init__(self, min_size, max_size, target_ratio):
        self.min_size = min_size
        self.max_size = max_size
        self.target_ratio = target_ratio

    def transform(self, sizes, X, y=None, meta=None):
        X, y = self._transform(sizes, X, y, meta)

        return {'X': X,
                'y': y
                }

    def load(self, filepath):
        return self

    def save(self, filepath):
        params = {}
        joblib.dump(params, filepath)

    def _transform(self, sizes, X, y=None, meta=None):
        raw_images = X[0]
        raw_images_adj = []
        for size, raw_image in tqdm(zip(sizes, raw_images)):
            h_adj, w_adj = self._get_adjusted_image_size(size, from_pil(raw_image))
            raw_image_adj = resize(from_pil(raw_image), (h_adj, w_adj), preserve_range=True).astype(np.uint8)
            raw_images_adj.append(to_pil(raw_image_adj))
        X_adj = [raw_images_adj]

        if y is not None and meta is not None:
            masks, contours, centers = y
            mask_dirnames = meta['file_path_masks'].tolist()

            masks_adj, contours_adj, centers_adj = [], [], []
            for size, mask, contour, center, mask_dirname in tqdm(zip(sizes, masks, contours, centers, mask_dirnames)):
                h_adj, w_adj = self._get_adjusted_image_size(size, from_pil(mask))

                mask_adj = resize(from_pil(mask), (h_adj, w_adj), preserve_range=True).astype(np.uint8)
                center_adj = resize(from_pil(center), (h_adj, w_adj), preserve_range=True).astype(np.uint8)
                contour_adj = self._get_contour(mask_dirname, (h_adj, w_adj))

                masks_adj.append(to_pil(mask_adj))
                contours_adj.append(to_pil(contour_adj))
                centers_adj.append(to_pil(center_adj))

            y_adj = [masks_adj, contours_adj, centers_adj]
        else:
            y_adj = None
        return X_adj, y_adj

    def _get_adjusted_image_size(self, mean_cell_size, img):
        h, w = img.shape[:2]
        img_area = h * w
        
        if mean_cell_size ==0:
            adj_ratio = 1.0
        else:
            size_ratio = img_area / mean_cell_size
            adj_ratio = size_ratio / self.target_ratio

        h_adj = int(clip(self.min_size, h * adj_ratio, self.max_size))
        w_adj = int(clip(self.min_size, w * adj_ratio, self.max_size))

        return h_adj, w_adj

    def _get_contour(self, mask_dirname, shape_adjusted):
        h_adj, w_adj = shape_adjusted
        overlayed_masks = np.zeros((h_adj, w_adj)).astype(np.uint8)
        for image_filepath in tqdm(glob.glob('{}/*'.format(mask_dirname))):
            image = np.asarray(Image.open(image_filepath))
            image = ndi.binary_fill_holes(image)
            image = resize(image, (h_adj, w_adj), preserve_range=True).astype(np.uint8)
            contour = get_contour(image)
            inside_contour = np.where(image & contour, 255, 0).astype(np.uint8)
            overlayed_masks += inside_contour
        overlayed_masks = np.where(overlayed_masks > 0, 255., 0.).astype(np.uint8)
        return overlayed_masks