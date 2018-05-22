import math
from itertools import product
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from attrdict import AttrDict
from torch.utils.data import Dataset, DataLoader
from sklearn.externals import joblib
from cv2 import resize

from augmentation import fast_seq, affine_seq, color_seq, patching_seq, padding_seq
from steps.base import BaseTransformer
from steps.pytorch.utils import ImgAug
from utils import from_pil, to_pil
from pipeline_config import MEAN, STD


class MetadataImageSegmentationDataset(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.train_mode = train_mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = image_augment
        self.image_augment_with_target = image_augment_with_target

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]
        Xi = self.load_image(img_filepath)

        if self.y is not None:
            mask_filepath = self.y[index]
            Mi = self.load_image(mask_filepath)

            if self.train_mode and self.image_augment_with_target is not None:
                Xi, Mi = from_pil(Xi, Mi)
                Xi, Mi = self.image_augment_with_target(Xi, Mi)
                if self.image_augment is not None:
                    Xi = self.image_augment(Xi)
                Xi, Mi = to_pil(Xi, Mi)

            if self.mask_transform is not None:
                Mi = self.mask_transform(Mi)

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi, Mi
        else:
            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi


class MetadataImageSegmentationTTA(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
        self.y = None

        self.train_mode = train_mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = image_augment
        self.image_augment_with_target = image_augment_with_target

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]
        Xi = self.load_image(img_filepath)

        Xi = from_pil(Xi)
        if self.image_augment is not None:
            Xi = self.image_augment(Xi)
        Xi = to_pil(Xi)

        if self.image_transform is not None:
            Xi = self.image_transform(Xi)
        return Xi


class MetadataImageSegmentationDatasetDistances(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.train_mode = train_mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = image_augment
        self.image_augment_with_target = image_augment_with_target

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')

    def load_joblib(selfself, filepath):
        return joblib.load(filepath)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]
        Xi = self.load_image(img_filepath)

        if self.y is not None:
            mask_filepath = self.y[index]
            Mi = self.load_image(mask_filepath)
            distance_filepath = mask_filepath.replace("/masks/", "/distances/")
            distance_filepath = os.path.splitext(distance_filepath)[0]
            size_filepath = distance_filepath.replace("/distances/", "/sizes/")
            Di = self.load_joblib(distance_filepath)
            Di = Di.astype(np.uint8)
            Si = self.load_joblib(size_filepath).astype(np.uint16)
            Si = np.sqrt(Si).astype(np.uint16)

            if self.train_mode and self.image_augment_with_target is not None:
                Xi, Mi = from_pil(Xi, Mi)
                Xi, Mi, Di, Si = self.image_augment_with_target(Xi, Mi, Di, Si)
                if self.image_augment is not None:
                    Xi = self.image_augment(Xi)
                Xi, Mi, Di, Si = to_pil(Xi, Mi, Di, Si)

            if not self.train_mode:
                Di = to_pil(Di)
                Si = to_pil(Si)

            if self.mask_transform is not None:
                Mi = self.mask_transform(Mi)
                Di = self.mask_transform(Di)
                Si = self.mask_transform(Si)
                Mi = torch.cat((Mi, Di, Si), dim=0)

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi, Mi
        else:
            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi


class ImageSegmentationDataset(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.train_mode = train_mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = image_augment
        self.image_augment_with_target = image_augment_with_target

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, index):
        Xi = self.X[0][index]

        if self.y is not None:
            Mi = self.y[0][index]

            if self.train_mode and self.image_augment_with_target is not None:
                Xi, Mi = from_pil(Xi, Mi)
                Xi, Mi = self.image_augment_with_target(Xi, Mi)
                if self.image_augment is not None:
                    Xi = self.image_augment(Xi)
                Xi, Mi = to_pil(Xi, Mi)

            if self.mask_transform is not None:
                Mi = self.mask_transform(Mi)

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi, Mi
        else:
            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi


class MetadataImageSegmentationMultitaskDataset(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.train_mode = train_mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = image_augment
        self.image_augment_with_target = image_augment_with_target

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]

        Xi = self.load_image(img_filepath)
        if self.y is not None:
            target_masks = []
            for i in range(y.shape[1]):
                filepath = self.y[index, i]
                mask = self.load_image(filepath)
                target_masks.append(mask)
            target_masks = [target[index] for target in self.y]
            data = [Xi] + target_masks

            if self.train_mode and self.image_augment_with_target is not None:
                data = from_pil(*data)
                data = self.image_augment_with_target(*data)
                if self.image_augment is not None:
                    data[0] = self.image_augment(data[0])
                data = to_pil(*data)

            if self.mask_transform is not None:
                data[1:] = [self.mask_transform(mask) for mask in data[1:]]

            if self.image_transform is not None:
                data[0] = self.image_transform(data[0])

            return data
        else:
            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi


class ImageSegmentationMultitaskDataset(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.train_mode = train_mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = image_augment
        self.image_augment_with_target = image_augment_with_target

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, index):
        Xi = self.X[0][index]

        if self.y is not None:
            target_masks = [target[index] for target in self.y]
            data = [Xi] + target_masks

            if self.train_mode and self.image_augment_with_target is not None:
                data = from_pil(*data)
                data = self.image_augment_with_target(*data)
                if self.image_augment is not None:
                    data[0] = self.image_augment(data[0])
                data = to_pil(*data)

            if self.mask_transform is not None:
                data[1:] = [self.mask_transform(mask) for mask in data[1:]]

            if self.image_transform is not None:
                data[0] = self.image_transform(data[0])

            return data
        else:
            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi


class ImageSegmentationLoaderBasic(BaseTransformer):
    def __init__(self, loader_params, dataset_params):
        super().__init__()
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.image_transform = transforms.Compose([transforms.Resize((self.dataset_params.h,
                                                                      self.dataset_params.w)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])
        self.mask_transform = transforms.Compose([transforms.Resize((self.dataset_params.h,
                                                                     self.dataset_params.w)),
                                                  transforms.Lambda(to_monochrome),
                                                  transforms.Lambda(to_tensor),
                                                  ])
        self.image_augment_with_target = ImgAug(fast_seq)
        self.image_augment = None

        self.dataset = None

    def transform(self, X, y, X_valid=None, y_valid=None, train_mode=True):
        if train_mode and y is not None:
            flow, steps = self.get_datagen(X, y, True, self.loader_params.training)
        else:
            flow, steps = self.get_datagen(X, None, False, self.loader_params.inference)

        if X_valid is not None and y_valid is not None:
            valid_flow, valid_steps = self.get_datagen(X_valid, y_valid, False, self.loader_params.inference)
        else:
            valid_flow = None
            valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, y, train_mode, loader_params):
        if train_mode:
            dataset = self.dataset(X, y,
                                   train_mode=True,
                                   image_augment=self.image_augment,
                                   image_augment_with_target=self.image_augment_with_target,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform)
        else:
            dataset = self.dataset(X, y,
                                   train_mode=False,
                                   image_augment=None,
                                   image_augment_with_target=None,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps


class ImageSegmentationLoaderInferencePadding(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_augment = ImgAug(padding_seq(pad_size=(self.dataset_params.h_pad,
                                                          self.dataset_params.w_pad),
                                                pad_method='replicate'
                                                ))
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])
        self.dataset = MetadataImageSegmentationTTA

    def transform(self, X, y, X_valid=None, y_valid=None, train_mode=False):
        flow, steps = self.get_datagen(X, None, False, self.loader_params.inference)
        valid_flow = None
        valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, y, train_mode, loader_params):
        dataset = self.dataset(X, None,
                               train_mode=False,
                               image_augment=self.image_augment,
                               image_augment_with_target=None,
                               mask_transform=None,
                               image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps


class MetadataImageSegmentationLoader(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = MetadataImageSegmentationDataset


class MetadataImageSegmentationLoaderDistances(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = MetadataImageSegmentationDatasetDistances

    def get_datagen(self, X, y, train_mode, loader_params):
        if train_mode:
            dataset = self.dataset(X, y,
                                   train_mode=True,
                                   image_augment=self.image_augment,
                                   image_augment_with_target=self.image_augment_with_target,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform)
        else:
            dataset = self.dataset(X, y,
                                   train_mode=False,
                                   image_augment=None,
                                   image_augment_with_target=None,
                                   mask_transform=None,
                                   image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps


class MetadataImageSegmentationMultitaskLoader(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = MetadataImageSegmentationMultitaskDataset


class ImageSegmentationLoader(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = ImageSegmentationDataset


class ImageSegmentationMultitaskLoader(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = ImageSegmentationMultitaskDataset


def binarize(x):
    x_ = x.convert('L')  # convert image to monochrome
    x_ = np.array(x_)
    x_ = (x_ > 125).astype(np.float32)
    return x_


def to_monochrome(x):
    x_ = x.convert('L')
    x_ = np.array(x_).astype(np.float32)  # convert image to monochrome
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_


def test_time_augmentation(img):
    for i in range(4):
        yield i * 90, np.rot90(img, i)


def generate_patches(img, patch_size, patch_stride):
    img_padded = get_mosaic_padded_image(img, patch_size, patch_stride)
    h_pad, w_pad = img_padded.shape[:2]

    h_patch_nr = math.ceil(h_pad / patch_stride) - math.floor(patch_size / patch_stride)
    w_patch_nr = math.ceil(w_pad / patch_stride) - math.floor(patch_size / patch_stride)

    for y_coordinate, x_coordinate in product(range(h_patch_nr), range(w_patch_nr)):
        if len(img.shape) == 2:
            img_patch = img_padded[y_coordinate * patch_stride:y_coordinate * patch_stride + patch_size,
                        x_coordinate * patch_stride:x_coordinate * patch_stride + patch_size]
        else:
            img_patch = img_padded[y_coordinate * patch_stride:y_coordinate * patch_stride + patch_size,
                        x_coordinate * patch_stride:x_coordinate * patch_stride + patch_size, :]
        yield y_coordinate, x_coordinate, img_patch


def get_mosaic_padded_image(img, patch_size, patch_stride):
    if len(img.shape) == 2:
        h_, w_ = img.shape
        c = 1
        img = np.expand_dims(img, axis=2)
        squeeze_output = True
    else:
        h_, w_, c = img.shape
        squeeze_output = False

    h, w = (max(h_, patch_size), max(w_, patch_size))
    if h > h_ or w > w_:
        img = resize(img, (h, w))

    h_pad, h_pad_top, h_pad_bottom, h_pad_end = get_padded_size(h, patch_size, patch_stride)
    w_pad, w_pad_left, w_pad_right, w_pad_end = get_padded_size(w, patch_size, patch_stride)

    img_padded = np.zeros((h_pad, w_pad, c))
    img_padded[h_pad_top:-h_pad_bottom, w_pad_left:-w_pad_right, :] = img

    img_padded[h_pad_top:-h_pad_bottom, :w_pad_left, :] = np.fliplr(img[:, :w_pad_left, :])
    img_padded[:h_pad_top, w_pad_left:-w_pad_right, :] = np.flipud(img[:h_pad_top, :, :])

    img_padded[h_pad_top:-h_pad_bottom, -w_pad_right:-w_pad_right + w_pad_end, :] = np.fliplr(
        img[:, -w_pad_right:-w_pad_right + w_pad_end, :])
    img_padded[-h_pad_bottom:-h_pad_bottom + h_pad_end, w_pad_left:-w_pad_right, :] = np.flipud(
        img[-h_pad_bottom:-h_pad_bottom + h_pad_end, :, :])

    if squeeze_output:
        img_padded = np.squeeze(img_padded)

    return img_padded


def get_padded_size(img_size, patch_size, patch_stride):
    min_image_size = img_size + 2 * patch_size
    for img_size_padded in range(img_size, 6 * img_size, 1):
        if (img_size_padded - patch_size) % patch_stride == 0 and img_size_padded >= min_image_size:
            break

    diff = img_size_padded - img_size
    pad_down, pad_up = patch_size, diff - patch_size
    if pad_up > patch_size and img_size < patch_size:
        pad_end = patch_size
    else:
        pad_end = pad_up
    return img_size_padded, pad_down, pad_up, pad_end
