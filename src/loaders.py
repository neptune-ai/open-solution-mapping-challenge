from itertools import product
import os

from attrdict import AttrDict
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.externals import joblib
from skimage.transform import rotate

from .augmentation import fast_seq, crop_seq, padding_seq
from .steps.base import BaseTransformer
from .steps.pytorch.utils import ImgAug
from .utils import from_pil, to_pil
from .pipeline_config import MEAN, STD


class MetadataImageSegmentationDataset(Dataset):
    def __init__(self, X, y,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

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

            if self.image_augment_with_target is not None:
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
    def __init__(self, X, tta_params,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
        self.tta_params = tta_params

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

        if self.tta_params is not None:
            tta_transform_specs = self.tta_params[index]
            Xi = test_time_augmentation_transform(Xi, tta_transform_specs)

        if self.image_augment is not None:
            Xi = self.image_augment(Xi)
        Xi = to_pil(Xi)

        if self.image_transform is not None:
            Xi = self.image_transform(Xi)

        return Xi


class MetadataImageSegmentationDatasetDistances(Dataset):
    def __init__(self, X, y,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

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
            Di = Di.astype(np.uint16)
            Si = self.load_joblib(size_filepath).astype(np.uint16)
            Si = np.sqrt(Si).astype(np.uint16)
            Xi, Mi = from_pil(Xi, Mi)
            if self.image_augment_with_target is not None:
                Xi, Mi, Di, Si = self.image_augment_with_target(Xi, Mi, Di, Si)
            if self.image_augment is not None:
                Xi = self.image_augment(Xi)
            Xi, Mi, Di, Si = to_pil(Xi, Mi, Di, Si)

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


class ImageSegmentationLoaderBasic(BaseTransformer):
    def __init__(self, loader_params, dataset_params):
        super().__init__()
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.image_transform = None
        self.mask_transform = None

        self.image_augment_with_target_train = None
        self.image_augment_with_target_inference = None
        self.image_augment_train = None
        self.image_augment_inference = None

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
                                   image_augment=self.image_augment_train,
                                   image_augment_with_target=self.image_augment_with_target_train,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform)
        else:
            dataset = self.dataset(X, y,
                                   image_augment=self.image_augment_inference,
                                   image_augment_with_target=self.image_augment_with_target_inference,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps


class MetadataImageSegmentationLoaderDistancesCropPad(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])
        self.mask_transform = transforms.Compose([transforms.Lambda(to_monochrome),
                                                  transforms.Lambda(to_tensor),
                                                  ])

        self.image_augment_with_target_train = ImgAug(crop_seq(crop_size=(self.dataset_params.h,
                                                                          self.dataset_params.w)))
        self.image_augment_with_target_inference = ImgAug(padding_seq(pad_size=(self.dataset_params.h_pad,
                                                                                self.dataset_params.w_pad),
                                                                      pad_method='replicate'
                                                                      ))

        self.dataset = MetadataImageSegmentationDatasetDistances


class MetadataImageSegmentationLoaderDistancesResize(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

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

        self.image_augment_with_target_train = ImgAug(fast_seq)

        self.dataset = MetadataImageSegmentationDatasetDistances


class MetadataImageSegmentationLoaderCropPad(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])
        self.mask_transform = transforms.Compose([transforms.Lambda(to_monochrome),
                                                  transforms.Lambda(to_tensor),
                                                  ])

        self.image_augment_with_target_train = ImgAug(crop_seq(crop_size=(self.dataset_params.h,
                                                                          self.dataset_params.w)))
        self.image_augment_with_target_inference = ImgAug(padding_seq(pad_size=(self.dataset_params.h_pad,
                                                                                self.dataset_params.w_pad),
                                                                      pad_method='replicate'
                                                                      ))

        self.dataset = MetadataImageSegmentationDataset


class MetadataImageSegmentationLoaderResize(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

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

        self.image_augment_with_target_train = ImgAug(fast_seq)

        self.dataset = MetadataImageSegmentationDataset


class ImageSegmentationLoaderInferencePadding(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_augment_inference = ImgAug(padding_seq(pad_size=(self.dataset_params.h_pad,
                                                                    self.dataset_params.w_pad),
                                                          pad_method='replicate'
                                                          ))
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])
        self.dataset = MetadataImageSegmentationTTA

    def transform(self, X, **kwargs):
        flow, steps = self.get_datagen(X, self.loader_params.inference)
        valid_flow = None
        valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, loader_params):
        dataset = self.dataset(X, None,
                               image_augment=self.image_augment_inference,
                               image_augment_with_target=None,
                               mask_transform=None,
                               image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps


class ImageSegmentationLoaderInferencePaddingTTA(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_augment_inference = ImgAug(padding_seq(pad_size=(self.dataset_params.h_pad,
                                                                    self.dataset_params.w_pad),
                                                          pad_method='replicate'
                                                          ))
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])
        self.dataset = MetadataImageSegmentationTTA

    def transform(self, X, tta_params, **kwargs):
        flow, steps = self.get_datagen(X, tta_params, self.loader_params.inference)
        valid_flow = None
        valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, tta_params, loader_params):
        dataset = self.dataset(X, tta_params,
                               image_augment=self.image_augment_inference,
                               image_augment_with_target=None,
                               mask_transform=None,
                               image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps


class TestTimeAugmentationGenerator(BaseTransformer):
    def __init__(self, **kwargs):
        self.tta_transformations = AttrDict(kwargs)

    def transform(self, X, **kwargs):
        X_tta_rows, tta_params, img_ids = [], [], []
        for i in range(len(X)):
            rows, params, ids = self._get_tta_data(i, X[i])
            tta_params.extend(params)
            img_ids.extend(ids)
            X_tta_rows.extend(rows)
        X_tta = pd.DataFrame(X_tta_rows)
        return {'X_tta': X_tta, 'tta_params': tta_params, 'img_ids': img_ids}

    def _get_tta_data(self, i, row):
        original_specs = {'ud_flip': False, 'lr_flip': False, 'rotation': 0}
        tta_specs = [original_specs]

        ud_options = [True, False] if self.tta_transformations.flip_ud else [False]
        lr_options = [True, False] if self.tta_transformations.flip_lr else [False]
        rot_options = [0, 90, 180, 270] if self.tta_transformations.rotation else [0]

        for ud, lr, rot in product(ud_options, lr_options, rot_options):
            if ud is False and lr is False and rot == 0:
                continue
            else:
                tta_specs.append({'ud_flip': ud, 'lr_flip': lr, 'rotation': rot})

        img_ids = [i] * len(tta_specs)
        X_rows = [row] * len(tta_specs)
        return X_rows, tta_specs, img_ids


class TestTimeAugmentationAggregator(BaseTransformer):
    def transform(self, images, tta_params, img_ids, **kwargs):
        averages_images = []
        for img_id in set(img_ids):
            tta_predictions_for_id = []
            for image, tta_param, ids in zip(images, tta_params, img_ids):
                if ids == img_id:
                    tta_prediction = test_time_augmentation_inverse_transform(image, tta_param)
                    tta_predictions_for_id.append(tta_prediction)
                else:
                    continue
            tta_averaged = np.mean(np.stack(tta_predictions_for_id, axis=-1), axis=-1)
            averages_images.append(tta_averaged)
        return {'aggregated_prediction': averages_images}


def test_time_augmentation_transform(image, tta_parameters):
    if tta_parameters['ud_flip']:
        image = np.flipud(image)
    elif tta_parameters['lr_flip']:
        image = np.fliplr(image)
    image = rotate(image, tta_parameters['rotation'], preserve_range=True)
    return image


def test_time_augmentation_inverse_transform(image, tta_parameters):
    image = per_channel_rotation(image.copy(), -1 * tta_parameters['rotation'])

    if tta_parameters['ud_flip']:
        image = per_channel_flipud(image.copy())
    elif tta_parameters['lr_flip']:
        image = per_channel_fliplr(image.copy())
    return image


def per_channel_flipud(x):
    x_ = x.copy()
    for i, channel in enumerate(x):
        x_[i, :, :] = np.flipud(channel)
    return x_


def per_channel_fliplr(x):
    x_ = x.copy()
    for i, channel in enumerate(x):
        x_[i, :, :] = np.fliplr(channel)
    return x_


def per_channel_rotation(x, angle):
    x_ = x.copy()
    for i, channel in enumerate(x):
        x_[i, :, :] = rotate(channel, angle, preserve_range=True)
    return x_


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
