import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from attrdict import AttrDict
from torch.utils.data import Dataset, DataLoader
from sklearn.externals import joblib

from augmentation import fast_seq, crop_seq, padding_seq
from steps.base import BaseTransformer
from steps.pytorch.utils import ImgAug
from utils import from_pil, to_pil
from pipeline_config import MEAN, STD


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
    def __init__(self, X, y,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
        super().__init__()
        self.X = X
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

        Xi = from_pil(Xi)
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
            if self.image_augment_with_target is not None:
                Xi, Mi = from_pil(Xi, Mi)
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

    def transform(self, X, y, X_valid=None, y_valid=None, train_mode=False):
        flow, steps = self.get_datagen(X, None, False, self.loader_params.inference)
        valid_flow = None
        valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, y, train_mode, loader_params):
        dataset = self.dataset(X, None,
                               image_augment=self.image_augment_inference,
                               image_augment_with_target=None,
                               mask_transform=None,
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
