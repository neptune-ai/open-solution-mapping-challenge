from math import ceil

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader

from steps.base import BaseTransformer


class MetadataImageDataset(Dataset):
    def __init__(self, X, y, image_transform, target_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.image_transform = image_transform
        self.image_augment = image_augment
        self.target_transform = target_transform

    def load_image(self, img_filepath):
        image = np.asarray(Image.open(img_filepath))
        image = image / 255.0
        return image

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]

        Xi = self.load_image(img_filepath)

        if self.image_augment is not None:
            Xi = self.image_augment(Xi)

        if self.image_transform is not None:
            Xi = self.image_transform(Xi)
        if self.y is not None:
            yi = self.y[index]
            if self.target_transform is not None:
                yi = self.target_transform(yi)
            return Xi, yi
        else:
            return Xi


class MetadataImageLoader(BaseTransformer):
    def __init__(self, loader_params):
        super().__init__()
        self.loader_params = loader_params

        self.dataset = MetadataImageDataset
        self.image_transform = transforms.ToTensor()
        self.target_transform = target_transform
        self.image_augment = None

    def transform(self, X, y, validation_data, train_mode):
        if train_mode:
            flow, steps = self.get_datagen(X, y, train_mode, self.loader_params['training'])
        else:
            flow, steps = self.get_datagen(X, y, train_mode, self.loader_params['inference'])

        if validation_data is not None:
            X_valid, y_valid = validation_data
            valid_flow, valid_steps = self.get_datagen(X_valid, y_valid, False, self.loader_params['inference'])
        else:
            valid_flow = None
            valid_steps = None

        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, y, train_mode, loader_params):
        if train_mode:
            dataset = self.dataset(X, y,
                                   image_augment=self.image_augment,
                                   image_transform=self.image_transform,
                                   target_transform=self.target_transform)

        else:
            dataset = self.dataset(X, y,
                                   image_augment=None,
                                   image_transform=self.image_transform,
                                   target_transform=self.target_transform)
        datagen = DataLoader(dataset, **loader_params)
        steps = ceil(X.shape[0] / loader_params['batch_size'])
        return datagen, steps

    def load(self, filepath):
        params = joblib.load(filepath)
        self.loader_params = params['loader_params']
        return self

    def save(self, filepath):
        params = {'loader_params': self.loader_params}
        joblib.dump(params, filepath)


def target_transform(y):
    return torch.from_numpy(y).type(torch.LongTensor)
