import numpy as np
from imgaug import augmenters as iaa

fast_seq = iaa.SomeOf((1, 2),
                      [iaa.Fliplr(0.5),
                       iaa.Flipud(0.5),
                       iaa.Affine(rotate=(0, 360),
                                  translate_percent=(-0.1, 0.1), mode='reflect'),
                       iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode='reflect')
                       ], random_order=True)

affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(rotate=(0, 360),
                           translate_percent=(-0.1, 0.1)),
                iaa.CropAndPad(percent=(-0.25, 0.25), pad_cval=0)
                ]),
    # Deformations
    iaa.PiecewiseAffine(scale=(0.00, 0.06))
], random_order=True)

color_seq = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.OneOf([iaa.AverageBlur(k=((5, 11), (5, 11))),
                                  iaa.AdditiveGaussianNoise(scale=0.05 * 255, per_channel=0.5)
                                  ]))
], random_order=True)

color_seq_RGB = iaa.Sequential([
    iaa.SomeOf((1, 2),
               [iaa.Sequential([
                   iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                   iaa.WithChannels(0, iaa.Add((0, 100))),
                   iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                   iaa.Sequential([
                       iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                       iaa.WithChannels(1, iaa.Add((0, 100))),
                       iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                   iaa.Sequential([
                       iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                       iaa.WithChannels(2, iaa.Add((0, 100))),
                       iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                   iaa.WithChannels(0, iaa.Add((0, 100))),
                   iaa.WithChannels(1, iaa.Add((0, 100))),
                   iaa.WithChannels(2, iaa.Add((0, 100)))]
               ),
    iaa.Sometimes(0.5, iaa.OneOf([iaa.AverageBlur(k=((5, 11), (5, 11))),
                                  iaa.AdditiveGaussianNoise(scale=0.05 * 255, per_channel=0.5)])
                  )
], random_order=True)


def patching_seq(crop_size):
    h, w = crop_size

    seq = iaa.Sequential([
        iaa.Affine(rotate=(0, 360)),
        CropFixed(px=h),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.1, 0.1), pad_cval=0)),
        iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.02, 0.06)))
    ], random_order=False)
    return seq


class CropFixed(iaa.Augmenter):
    def __init__(self, px=None, name=None, deterministic=False, random_state=None):
        super(CropFixed, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.px = px

    def _augment_images(self, images, random_state, parents, hooks):

        result = []
        seeds = random_state.randint(0, 10 ** 6, (len(images),))
        for i, image in enumerate(images):
            seed = seeds[i]
            image_cr = self._random_crop_or_pad(seed, image)
            result.append(image_cr)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _random_crop_or_pad(self, seed, image):
        height, width = image.shape[:2]

        if height <= self.px and width > self.px:
            image_processed = self._random_crop(seed, image, crop_h=False, crop_w=True)
            image_processed = self._pad(image_processed)
        elif height > self.px and width <= self.px:
            image_processed = self._random_crop(seed, image, crop_h=True, crop_w=False)
            image_processed = self._pad(image_processed)
        elif height <= self.px and width <= self.px:
            image_processed = self._pad(image)
        else:
            image_processed = self._random_crop(seed, image, crop_h=True, crop_w=True)
        return image_processed

    def _random_crop(self, seed, image, crop_h=True, crop_w=True):
        height, width = image.shape[:2]

        if crop_h:
            np.random.seed(seed)
            crop_top = np.random.randint(height - self.px)
            crop_bottom = crop_top + self.px
        else:
            crop_top, crop_bottom = (0, height)

        if crop_w:
            np.random.seed(seed + 1)
            crop_left = np.random.randint(width - self.px)
            crop_right = crop_left + self.px
        else:
            crop_left, crop_right = (0, width)

        if len(image.shape) == 2:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right]
        else:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right, :]
        return image_cropped

    def _pad(self, image):
        if len(image.shape) == 2:
            height, width = image.shape
            image_padded = np.zeros((max(height, self.px), max(width, self.px))).astype(np.uint8)
            image_padded[:height, :width] = image
        else:
            height, width, channels = image.shape
            image_padded = np.zeros((max(height, self.px), max(width, self.px), channels)).astype(np.uint8)
            image_padded[:height, :width, :] = image
        return image_padded

    def get_parameters(self):
        return []
