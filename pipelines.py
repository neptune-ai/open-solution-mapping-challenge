from functools import partial

import loaders
from steps.base import Step, Dummy
from steps.preprocessing.misc import XYSplit
from utils import squeeze_inputs
from models import PyTorchUNet, PyTorchUNetStream, PyTorchUNetWeighted, PyTorchUNetWeightedStream
from postprocessing import Resizer, CategoryMapper, MulticlassLabeler, DenseCRF, \
    ScoreBuilder, MaskEroder, LabeledMaskDilator, \
    ResizerStream, CategoryMapperStream, MulticlassLabelerStream, DenseCRFStream, \
    MaskEroderStream, LabeledMaskDilatorStream


def unet(config, train_mode):
    if train_mode:
        save_output = False
        load_saved_output = False
    else:
        save_output = False
        load_saved_output = False

    loader = preprocessing(config, model_type='single', is_train=train_mode)
    unet = Step(name='unet',
                transformer=PyTorchUNetStream(**config.unet) if config.execution.stream_mode else PyTorchUNet(
                    **config.unet),
                input_steps=[loader],
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output, load_saved_output=load_saved_output)

    mask_postprocessed = mask_postprocessing(loader, unet, config, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'images')]),
                           'y_scores': ([(mask_postprocessed.name, 'scores')])
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=save_output,
                  load_saved_output=False)
    return output


def unet_weighted(config, train_mode):
    unet_weighted = unet(config, train_mode)
    unet_weighted.get_step("loader").transformer = loaders.MetadataImageSegmentationLoaderDistances(**config.loader)
    unet_weighted.get_step("unet").transformer = PyTorchUNetWeightedStream(
        **config.unet) if config.execution.stream_mode else PyTorchUNetWeighted(
        **config.unet)
    return unet_weighted


def preprocessing(config, model_type, is_train, loader_mode=None):
    if model_type == 'single':
        loader = _preprocessing_single_generator(config, is_train, loader_mode)
    elif model_type == 'multitask':
        loader = _preprocessing_multitask_generator(config, is_train, loader_mode)
    else:
        raise NotImplementedError
    return loader


def multiclass_object_labeler(postprocessed_mask, config, save_output=True):
    labeler = Step(name='labeler',
                   transformer=MulticlassLabelerStream() if config.execution.stream_mode else MulticlassLabeler(),
                   input_steps=[postprocessed_mask],
                   adapter={'images': ([(postprocessed_mask.name, 'eroded_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath,
                   save_output=save_output)
    return labeler


def _preprocessing_single_generator(config, is_train, use_patching):
    if use_patching:
        raise NotImplementedError
    else:
        if is_train:
            xy_train = Step(name='xy_train',
                            transformer=XYSplit(**config.xy_splitter),
                            input_data=['input'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('input', 'train_mode')])
                                     },
                            cache_dirpath=config.env.cache_dirpath)

            xy_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.xy_splitter),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta_valid')]),
                                         'train_mode': ([('input', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)

            loader = Step(name='loader',
                          transformer=loaders.MetadataImageSegmentationLoader(**config.loader),
                          input_data=['input'],
                          input_steps=[xy_train, xy_inference],
                          adapter={'X': ([('xy_train', 'X')], squeeze_inputs),
                                   'y': ([('xy_train', 'y')], squeeze_inputs),
                                   'train_mode': ([('input', 'train_mode')]),
                                   'X_valid': ([('xy_inference', 'X')], squeeze_inputs),
                                   'y_valid': ([('xy_inference', 'y')], squeeze_inputs),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
        else:
            xy_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.xy_splitter),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('input', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)

            loader = Step(name='loader',
                          transformer=loaders.MetadataImageSegmentationLoader(**config.loader),
                          input_data=['input'],
                          input_steps=[xy_inference, xy_inference],
                          adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                                   'y': ([('xy_inference', 'y')], squeeze_inputs),
                                   'train_mode': ([('input', 'train_mode')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    return loader


def _preprocessing_multitask_generator(config, is_train, use_patching):
    if use_patching:
        raise NotImplementedError
    else:
        if is_train:
            xy_train = Step(name='xy_train',
                            transformer=XYSplit(**config.xy_splitter_multitask),
                            input_data=['input'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('input', 'train_mode')])
                                     },
                            cache_dirpath=config.env.cache_dirpath)

            xy_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.splitter_config),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta_valid')]),
                                         'train_mode': ([('input', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)

            loader = Step(name='loader',
                          transformer=loaders.MetadataImageSegmentationMultitaskLoader(**config.loader),
                          input_data=['input'],
                          input_steps=[xy_train, xy_inference],
                          adapter={'X': ([('xy_train', 'X')], squeeze_inputs),
                                   'y': ([('xy_train', 'y')]),
                                   'train_mode': ([('input', 'train_mode')]),
                                   'X_valid': ([('xy_inference', 'X')], squeeze_inputs),
                                   'y_valid': ([('xy_inference', 'y')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
        else:
            xy_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.xy_splitter_multitask),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('input', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)

            loader = Step(name='loader',
                          transformer=loaders.MetadataImageSegmentationMultitaskLoader(**config.loader),
                          input_data=['input'],
                          input_steps=[xy_inference, xy_inference],
                          adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                                   'y': ([('xy_inference', 'y')], squeeze_inputs),
                                   'train_mode': ([('input', 'train_mode')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    return loader


def mask_postprocessing(loader, model, config, save_output=False):
    if config.postprocessor.crf.apply_crf:
        dense_crf = Step(name='dense_crf',
                         transformer=DenseCRFStream(**config.postprocessor.crf) if config.execution.stream_mode \
                             else DenseCRF(**config.postprocessor.crf),
                         input_steps=[loader, model],
                         adapter={'images': ([(model.name, 'multichannel_map_prediction')]),
                                  'raw_images_generator': ([(loader.name, 'datagen')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

        mask_resize = Step(name='mask_resize',
                           transformer=ResizerStream() if config.execution.stream_mode else Resizer(),
                           input_data=['input'],
                           input_steps=[dense_crf],
                           adapter={'images': ([('dense_crf', 'crf_images')]),
                                    'target_sizes': ([('input', 'target_sizes')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)
    else:
        mask_resize = Step(name='mask_resize',
                           transformer=ResizerStream() if config.execution.stream_mode else Resizer(),
                           input_data=['input'],
                           input_steps=[model],
                           adapter={'images': ([(model.name, 'multichannel_map_prediction')]),
                                    'target_sizes': ([('input', 'target_sizes')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    category_mapper = Step(name='category_mapper',
                           transformer=CategoryMapperStream() if config.execution.stream_mode else CategoryMapper(),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize', 'resized_images')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    mask_erosion = Step(name='mask_erosion',
                        transformer=MaskEroderStream(**config.postprocessor) if config.execution.stream_mode
                        else MaskEroder(**config.postprocessor),
                        input_steps=[category_mapper],
                        adapter={'images': ([(category_mapper.name, 'categorized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        load_saved_output=False)
        
    detached = multiclass_object_labeler(mask_erosion, config, save_output=save_output)

    mask_dilation = Step(name='mask_dilation',
                         transformer=LabeledMaskDilatorStream(**config.postprocessor) if config.execution.stream_mode
                         else LabeledMaskDilator(**config.postprocessor),
                         input_steps=[detached],
                         adapter={'images': ([(detached.name, 'labeled_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         load_saved_output=False)

    score_builder = Step(name='score_builder',
                         transformer=ScoreBuilder(),
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
             }
