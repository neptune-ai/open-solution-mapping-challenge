from functools import partial

from . import loaders
from .steps.base import Step, Dummy
from .steps.preprocessing.misc import XYSplit
from .utils import squeeze_inputs, make_apply_transformer, make_apply_transformer_stream
from .models import PyTorchUNet, PyTorchUNetWeighted, PyTorchUNetStream, PyTorchUNetWeightedStream, ScoringLightGBM
from . import postprocessing as post


def unet(config, train_mode):
    save_output = False
    load_saved_output = False

    make_apply_transformer_ = make_apply_transformer_stream if config.execution.stream_mode else make_apply_transformer

    loader = preprocessing_generator(config, is_train=train_mode)
    unet = Step(name='unet',
                transformer=PyTorchUNetStream(**config.unet) if config.execution.stream_mode
                else PyTorchUNet(**config.unet),
                input_data=['callback_input'],
                input_steps=[loader],
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output, load_saved_output=load_saved_output)

    mask_postprocessed = mask_postprocessing(unet, config, make_apply_transformer_, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'images_with_scores')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=save_output, load_saved_output=load_saved_output)
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

    make_apply_transformer_ = make_apply_transformer_stream if config.execution.stream_mode else make_apply_transformer

    unet_pipeline = unet(config, train_mode=False).get_step('unet')

    loader = unet_pipeline.get_step("loader")
    loader.transformer = loaders.ImageSegmentationLoaderInferencePadding(**config.loader)

    prediction_crop = Step(name='prediction_crop',
                           transformer=make_apply_transformer_(partial(post.crop_image_center_per_class,
                                                                       **config.postprocessor.prediction_crop),
                                                               output_name='cropped_images'),
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
    mask_postprocessed = mask_postprocessing(prediction_renamed, config, make_apply_transformer_, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'images_with_scores')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=save_output)
    return output


def unet_tta(config):
    if config.execution.stream_mode:
        raise Exception('TTA not available in stream mode')
    save_output = False

    loader, tta_generator = preprocessing_generator_tta(config)
    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.unet),
                input_steps=[loader],
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output)

    tta_aggregator = Step(name='tta_aggregator',
                          transformer=loaders.TestTimeAugmentationAggregator(**config.tta_aggregator),
                          input_steps=[unet, tta_generator],
                          adapter={'images': ([(unet.name, 'multichannel_map_prediction')]),
                                   'tta_params': ([(tta_generator.name, 'tta_params')]),
                                   'img_ids': ([(tta_generator.name, 'img_ids')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=save_output)

    if config.execution.loader_mode == 'crop_and_pad':
        prediction_crop = Step(name='prediction_crop',
                               transformer=make_apply_transformer(partial(post.crop_image_center_per_class,
                                                                          **config.postprocessor.prediction_crop),
                                                                  output_name='cropped_images'),
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
    elif config.execution.loader_mode == 'resize':
        prediction_renamed = Step(name='prediction_renamed',
                                  transformer=Dummy(),
                                  input_steps=[tta_aggregator],
                                  adapter={
                                      'multichannel_map_prediction': (
                                          [(tta_aggregator.name, 'aggregated_prediction')]), },
                                  cache_dirpath=config.env.cache_dirpath,
                                  save_output=save_output)
    else:
        raise NotImplementedError('only crop_and_pad and resize options available')

    mask_postprocessed = mask_postprocessing(prediction_renamed, config, make_apply_transformer, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'images_with_scores')]),
                           },
                  cache_dirpath=config.env.cache_dirpath, save_output=save_output)
    return output


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


def preprocessing_generator_tta(config):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderInferencePaddingTTA
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResizeTTA
    else:
        raise NotImplementedError('only crop_and_pad and resize options available')

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
                  transformer=Loader(**config.loader),
                  input_steps=[xy_inference, tta_generator],
                  adapter={'X': ([(tta_generator.name, 'X_tta')], squeeze_inputs),
                           'tta_params': ([(tta_generator.name, 'tta_params')], squeeze_inputs),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return loader, tta_generator


def mask_postprocessing(model, config, make_transformer, **kwargs):
    mask_resize = Step(name='mask_resize',
                       transformer=make_transformer(post.resize_image,
                                                    output_name='resized_images',
                                                    apply_on=['images', 'target_sizes']),
                       input_data=['input'],
                       input_steps=[model],
                       adapter={'images': ([(model.name, 'multichannel_map_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       cache_output=not config.execution.stream_mode, **kwargs)

    category_mapper = Step(name='category_mapper',
                           transformer=make_transformer(post.categorize_image,
                                                        output_name='categorized_images'),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize', 'resized_images')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath, **kwargs)

    mask_erosion = Step(name='mask_erosion',
                        transformer=make_transformer(partial(post.erode_image, **config.postprocessor.mask_erosion),
                                                     output_name='eroded_images'),
                        input_steps=[category_mapper],
                        adapter={'images': ([(category_mapper.name, 'categorized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath, **kwargs)

    labeler = Step(name='labeler',
                   transformer=make_transformer(post.label_multiclass_image,
                                                output_name='labeled_images'),
                   input_steps=[mask_erosion],
                   adapter={'images': ([(mask_erosion.name, 'eroded_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath, **kwargs)

    mask_dilation = Step(name='mask_dilation',
                         transformer=make_transformer(partial(post.dilate_image, **config.postprocessor.mask_dilation),
                                                      output_name='dilated_images'),
                         input_steps=[labeler],
                         adapter={'images': ([(labeler.name, 'labeled_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath, **kwargs)

    score_builder = Step(name='score_builder',
                         transformer=make_transformer(post.build_score,
                                                      output_name='images_with_scores',
                                                      apply_on=['images', 'probabilities']),
                         input_steps=[mask_dilation, mask_resize],
                         adapter={'images': ([(mask_dilation.name, 'dilated_images')]),
                                  'probabilities': ([(mask_resize.name, 'resized_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath, **kwargs)

    return score_builder


def lgbm_train(config):

    save_output = False
    unet_type = 'weighted'

    if unet_type=='standard':
        unet_pipeline = unet(config, train_mode=False)
    elif unet_type=='weighted':
        unet_pipeline = unet_weighted(config, train_mode=False)
    else:
        raise NotImplementedError

    mask_dilation = unet_pipeline.get_step('mask_dilation')
    mask_resize = unet_pipeline.get_step('mask_resize')

    mask_dilation.transformer = post.LabeledMaskDilatorStream(**config.postprocessor.mask_dilation)
    mask_resize.transformer = post.ResizerStream()
    if config.postprocessor.crf.apply_crf:
        unet_pipeline.get_step('dense_crf').transformer = post.DenseCRFStream(**config.postprocessor.crf)
    unet_pipeline.get_step('category_mapper').transformer = post.CategoryMapperStream()
    unet_pipeline.get_step('mask_erosion').transformer = post.MaskEroderStream(**config.postprocessor.mask_erosion)
    unet_pipeline.get_step('labeler').transformer = post.MulticlassLabelerStream()

    feature_extractor = Step(name='feature_extractor',
                             transformer=post.FeatureExtractor(**config['postprocessor']['feature_extractor']),
                             input_steps=[mask_dilation, mask_resize],
                             input_data=['input', 'specs'],
                             adapter={'images': ([(mask_dilation.name, 'dilated_images')]),
                                      'probabilities': ([(mask_resize.name, 'resized_images')]),
                                      'annotations': ([('input', 'annotations')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath,
                             save_output=True,
                             load_saved_output=False)

    scoring_model = Step(name='scoring_model',
                         transformer=ScoringLightGBM(**config['postprocessor']['lightGBM']),
                         input_steps=[feature_extractor],
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output,
                         force_fitting=True#test
                         )

    return scoring_model


def lgbm_inference(config, input_pipeline):

    save_output=False

    mask_dilation = input_pipeline(config, train_mode=False).get_step('mask_dilation')
    mask_resize = input_pipeline(config, train_mode=False).get_step('mask_resize')

    feature_extractor = Step(name='feature_extractor',
                             transformer=post.FeatureExtractor(),
                             input_steps=[mask_dilation, mask_resize],
                             input_data=['input'],
                             adapter={'images': ([(mask_dilation.name, 'dilated_images')]),
                                      'probabilities': ([(mask_resize.name, 'resized_images')]),
                                      'annotations': ([('input', 'annotations')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath,
                             save_output=save_output)

    scoring_model = Step(name='scoring_model',
                         transformer=ScoringLightGBM(**config['postprocessor']['lightGBM']),
                         input_steps=[feature_extractor],
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    score_builder = Step(name='score_builder',
                         transformer=post.ScoreImageJoiner(),
                         input_steps=[scoring_model, mask_dilation],
                         adapter={'images': ([(mask_dilation.name, 'dilated_images')]),
                                  'scores': ([(scoring_model.name, 'scores')])},
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    nms = Step(name='nms',
               transformer=post.NonMaximumSupression(**config['postprocessor']['nms']),
               input_steps=[score_builder],
               cache_dirpath=config.env.cache_dirpath,
               save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[nms],
                  adapter={'y_pred': ([(nms.name, 'images_with_scores')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=save_output,
                  load_saved_output=False)
    return output


PIPELINES = {'unet': {'train': partial(unet, train_mode=True),
                      'inference': partial(unet, train_mode=False),
                      },
             'unet_weighted': {'train': partial(unet_weighted, train_mode=True),
                               'inference': partial(unet_weighted, train_mode=False),
                               },
             'unet_tta': {'inference': unet_tta,
                          },
             'unet_padded': {'inference': unet_padded,
                             },
             'lgbm': {'train': lgbm_train},
             'unet_lgbm': {'inference': partial(lgbm_inference, input_pipeline=partial(unet, train_mode=False))},
             'unet_padded_lgbm': {'inference': partial(lgbm_inference, input_pipeline=unet_padded)},

             }
