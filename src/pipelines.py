from functools import partial
import os

from . import loaders
from .steps.base import Step, Dummy
from .steps.preprocessing.misc import XYSplit
from .utils import squeeze_inputs
from .models import PyTorchUNet, PyTorchUNetStream, PyTorchUNetWeighted, PyTorchUNetWeightedStream
from . import postprocessing as post


def unet(config, train_mode):
    save_output = False
    load_saved_output = False

    loader = preprocessing_generator(config, is_train=train_mode)
    unet = Step(name='unet',
                transformer=PyTorchUNetStream(**config.unet)
                if config.execution.stream_mode
                else PyTorchUNet(**config.unet),
                input_data=['callback_input'],
                input_steps=[loader],
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output, load_saved_output=load_saved_output)

    mask_postprocessed = mask_postprocessing(loader, unet, config, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'images_with_scores')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=save_output,
                  load_saved_output=False)
    return output


def unet_weighted(config, train_mode):
    unet_weighted = unet(config, train_mode)
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.MetadataImageSegmentationLoaderDistancesCropPad
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.MetadataImageSegmentationLoaderDistancesResize
    else:
        raise NotImplementedError('only crop_and_pad and resize options available')
    unet_weighted.get_step("loader").transformer = Loader(**config.loader)
    unet_weighted.get_step("unet").transformer = PyTorchUNetWeightedStream(**config.unet) \
        if config.execution.stream_mode else PyTorchUNetWeighted(**config.unet)
    return unet_weighted


def unet_padded(config):
    save_output = False

    unet_pipeline = unet(config, train_mode=False).get_step('unet')

    loader = unet_pipeline.get_step("loader")
    loader.transformer = loaders.ImageSegmentationLoaderInferencePadding(**config.loader)

    prediction_crop = Step(name='prediction_crop',
                           transformer=post.PredictionCropStream(
                               **config.postprocessor.prediction_crop) if config.execution.stream_mode \
                               else post.PredictionCrop(**config.postprocessor.prediction_crop),
                           input_steps=[unet_pipeline],
                           adapter={'images': ([(unet_pipeline.name, 'multichannel_map_prediction')]), },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    prediction_renamed = Step(name='prediction_renamed',
                              transformer=Dummy(),
                              input_steps=[prediction_crop],
                              adapter={
                                  'multichannel_map_prediction': ([(prediction_crop.name, 'cropped_images')]), },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)
    mask_postprocessed = mask_postprocessing(loader, prediction_renamed, config, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'images_with_scores')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=save_output)
    return output


def unet_padded_tta(config):
    save_output = False

    if config.execution.stream_mode:
        raise NotImplementedError('TTA not available in the stream mode')

    loader, tta_generator = preprocessing_generator_padded_tta(config)
    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.unet),
                input_steps=[loader],
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output)

    tta_aggregator = Step(name='tta_aggregator',
                          transformer=loaders.TestTimeAugmentationAggregator(),
                          input_steps=[unet, tta_generator],
                          adapter={'images': ([(unet.name, 'multichannel_map_prediction')]),
                                   'tta_params': ([(tta_generator.name, 'tta_params')]),
                                   'img_ids': ([(tta_generator.name, 'img_ids')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=save_output)

    prediction_crop = Step(name='prediction_crop',
                           transformer=post.PredictionCrop(**config.postprocessor.prediction_crop),
                           input_steps=[tta_aggregator],
                           adapter={'images': ([(tta_aggregator.name, 'aggregated_prediction')]), },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    prediction_renamed = Step(name='prediction_renamed',
                              transformer=Dummy(),
                              input_steps=[prediction_crop],
                              adapter={
                                  'multichannel_map_prediction': ([(prediction_crop.name, 'cropped_images')]), },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)
    mask_postprocessed = mask_postprocessing(loader, prediction_renamed, config, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'images_with_scores')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=save_output)
    return output


def multiclass_object_labeler(postprocessed_mask, config, save_output=False):
    labeler = Step(name='labeler',
                   transformer=post.MulticlassLabelerStream() if config.execution.stream_mode else post.MulticlassLabeler(),
                   input_steps=[postprocessed_mask],
                   adapter={'images': ([(postprocessed_mask.name, 'eroded_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath,
                   save_output=save_output)
    return labeler


def preprocessing_generator(config, is_train):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.MetadataImageSegmentationLoaderCropPad
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.MetadataImageSegmentationLoaderResize
    else:
        raise NotImplementedError('only crop_and_pad and resize options available')

    if is_train:
        xy_train = Step(name='xy_train',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input', 'specs'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'train_mode': ([('specs', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

        xy_inference = Step(name='xy_inference',
                            transformer=XYSplit(**config.xy_splitter),
                            input_data=['callback_input', 'specs'],
                            adapter={'meta': ([('callback_input', 'meta_valid')]),
                                     'train_mode': ([('specs', 'train_mode')])
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        loader = Step(name='loader',
                      transformer=Loader(**config.loader),
                      input_data=['specs'],
                      input_steps=[xy_train, xy_inference],
                      adapter={'X': ([('xy_train', 'X')], squeeze_inputs),
                               'y': ([('xy_train', 'y')], squeeze_inputs),
                               'train_mode': ([('specs', 'train_mode')]),
                               'X_valid': ([('xy_inference', 'X')], squeeze_inputs),
                               'y_valid': ([('xy_inference', 'y')], squeeze_inputs),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    else:
        xy_inference = Step(name='xy_inference',
                            transformer=XYSplit(**config.xy_splitter),
                            input_data=['input', 'specs'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('specs', 'train_mode')])
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        loader = Step(name='loader',
                      transformer=Loader(**config.loader),
                      input_data=['specs'],
                      input_steps=[xy_inference, xy_inference],
                      adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                               'y': ([('xy_inference', 'y')], squeeze_inputs),
                               'train_mode': ([('specs', 'train_mode')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return loader


def preprocessing_generator_padded_tta(config):
    xy_inference = Step(name='xy_inference',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input', 'specs'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'train_mode': ([('specs', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    tta_generator = Step(name='tta_generator',
                         transformer=loaders.TestTimeAugmentationGenerator(**config.tta_generator),
                         input_steps=[xy_inference],
                         adapter={'X': ([('xy_inference', 'X')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)

    loader = Step(name='loader',
                  transformer=loaders.ImageSegmentationLoaderInferencePaddingTTA(**config.loader),
                  input_steps=[xy_inference, tta_generator],
                  adapter={'X': ([(tta_generator.name, 'X_tta')], squeeze_inputs),
                           'tta_params': ([(tta_generator.name, 'tta_params')], squeeze_inputs),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return loader, tta_generator


def mask_postprocessing(loader, model, config, save_output=False):
    if config.postprocessor.crf.apply_crf:
        dense_crf = Step(name='dense_crf',
                         transformer=post.DenseCRFStream(**config.postprocessor.crf) if config.execution.stream_mode \
                             else post.DenseCRF(**config.postprocessor.crf),
                         input_steps=[loader, model],
                         adapter={'images': ([(model.name, 'multichannel_map_prediction')]),
                                  'raw_images_generator': ([(loader.name, 'datagen')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

        mask_resize = Step(name='mask_resize',
                           transformer=post.ResizerStream() if config.execution.stream_mode else post.Resizer(),
                           input_data=['input'],
                           input_steps=[dense_crf],
                           adapter={'images': ([('dense_crf', 'crf_images')]),
                                    'target_sizes': ([('input', 'target_sizes')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)
    else:
        mask_resize = Step(name='mask_resize',
                           transformer=post.ResizerStream() if config.execution.stream_mode else post.Resizer(),
                           input_data=['input'],
                           input_steps=[model],
                           adapter={'images': ([(model.name, 'multichannel_map_prediction')]),
                                    'target_sizes': ([('input', 'target_sizes')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    category_mapper = Step(name='category_mapper',
                           transformer=post.CategoryMapperStream() if config.execution.stream_mode else post.CategoryMapper(),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize', 'resized_images')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    mask_erosion = Step(name='mask_erosion',
                        transformer=post.MaskEroderStream(
                            **config.postprocessor.mask_erosion) if config.execution.stream_mode
                        else post.MaskEroder(**config.postprocessor.mask_erosion),
                        input_steps=[category_mapper],
                        adapter={'images': ([(category_mapper.name, 'categorized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output)

    detached = multiclass_object_labeler(mask_erosion, config, save_output=save_output)

    mask_dilation = Step(name='mask_dilation',
                         transformer=post.LabeledMaskDilatorStream(
                             **config.postprocessor.mask_dilation) if config.execution.stream_mode
                         else post.LabeledMaskDilator(**config.postprocessor.mask_dilation),
                         input_steps=[detached],
                         adapter={'images': ([(detached.name, 'labeled_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath, save_output=save_output)

    score_builder = Step(name='score_builder',
                         transformer=post.ScoreBuilder(),
                         input_steps=[mask_dilation, mask_resize],
                         adapter={'images': ([(mask_dilation.name, 'dilated_images')]),
                                  'probabilities': ([(mask_resize.name, 'resized_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    return score_builder


PIPELINES = {'unet': {'train': partial(unet, train_mode=True),
                      'inference': partial(unet, train_mode=False),
                      },
             'unet_weighted': {'train': partial(unet_weighted, train_mode=True),
                               'inference': partial(unet_weighted, train_mode=False),
                               },
             'unet_padded': {'inference': unet_padded,
                             },
             'unet_padded_tta': {'inference': unet_padded_tta,
                                 },

             }
