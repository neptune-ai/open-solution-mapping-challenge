import cv2
import numpy as np
from imgaug import augmenters as iaa


class PadFixed(iaa.Augmenter):
    PAD_FUNCTION = {'reflect': cv2.BORDER_REFLECT_101,
                    'replicate': cv2.BORDER_REPLICATE,
                    }

    def __init__(self, pad=None, pad_method=None, name=None, deterministic=False, random_state=None):
        super().__init__(name, deterministic, random_state)
        self.pad = pad
        self.pad_method = pad_method

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        for i, image in enumerate(images):
            image_pad = self._reflect_pad(image)
            result.append(image_pad)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _reflect_pad(self, img):
        h_pad, w_pad = self.pad
        img_padded = cv2.copyMakeBorder(img.copy(), h_pad, h_pad, w_pad, w_pad,
                                        PadFixed.PAD_FUNCTION[self.pad_method])
        return img_padded

    def get_parameters(self):
        return []


class RandomCropFixedSize(iaa.Augmenter):
    def __init__(self, px=None, name=None, deterministic=False, random_state=None):
        super(RandomCropFixedSize, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.px = px
        if isinstance(self.px, tuple):
            self.px_h, self.px_w = self.px
        elif isinstance(self.px, int):
            self.px_h = self.px
            self.px_w = self.px
        else:
            raise NotImplementedError

    def _augment_images(self, images, random_state, parents, hooks):

        result = []
        seeds = random_state.randint(0, 10 ** 6, (len(images),))
        for i, image in enumerate(images):
            seed = seeds[i]
            image_cr = self._random_crop(seed, image)
            result.append(image_cr)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _random_crop(self, seed, image):
        height, width = image.shape[:2]

        np.random.seed(seed)
        crop_top = np.random.randint(height - self.px_h)
        crop_bottom = crop_top + self.px_h

        np.random.seed(seed + 1)
        crop_left = np.random.randint(width - self.px_w)
        crop_right = crop_left + self.px_w

        if len(image.shape) == 2:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right]
        else:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right, :]
        return image_cropped

    def get_parameters(self):
        return []


fast_seq = iaa.SomeOf((1, 2),
                      [iaa.Fliplr(0.5),
                       iaa.Flipud(0.5),
                       iaa.Affine(rotate=(0, 360),
                                  translate_percent=(-0.1, 0.1), mode='reflect'),
                       ], random_order=True)

# def crop_seq(crop_size):
#     return RandomCropFixedSize(px=crop_size)

crop_seq = RandomCropFixedSize(px=(288, 288))


def padding_seq(pad_size, pad_method):
    seq = iaa.Sequential([PadFixed(pad=pad_size, pad_method=pad_method),
                          ]).to_deterministic()
    return seq
