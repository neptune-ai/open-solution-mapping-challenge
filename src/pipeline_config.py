import os

from attrdict import AttrDict
from deepsense import neptune

from .utils import read_params

ctx = neptune.Context()
params = read_params(ctx, fallback_file='neptune.yaml')

SIZE_COLUMNS = ['height', 'width']
X_COLUMNS = ['file_path_image']
Y_COLUMNS = ['file_path_mask_eroded_0_dilated_0']
Y_COLUMNS_SCORING = ['ImageId']
CATEGORY_IDS = [None, 100]
SEED = 1234
CATEGORY_LAYERS = [1, 19]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

GLOBAL_CONFIG = {'exp_root': params.experiment_dir,
                 'load_in_memory': params.load_in_memory,
                 'num_workers': params.num_workers,
                 'num_classes': 2,
                 'img_H-W': (params.image_h, params.image_w),
                 'batch_size_train': params.batch_size_train,
                 'batch_size_inference': params.batch_size_inference,
                 'loader_mode': params.loader_mode,
                 'stream_mode': params.stream_mode
                 }

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir},
    'execution': GLOBAL_CONFIG,
    'xy_splitter': {'x_columns': X_COLUMNS,
                    'y_columns': Y_COLUMNS,
                    },
    'reader_single': {'x_columns': X_COLUMNS,
                      'y_columns': Y_COLUMNS,
                      },
    'loader': {'dataset_params': {'h_pad': params.h_pad,
                                  'w_pad': params.w_pad,
                                  'h': params.image_h,
                                  'w': params.image_w,
                                  'pad_method': params.pad_method
                                  },
               'loader_params': {'training': {'batch_size': params.batch_size_train,
                                              'shuffle': True,
                                              'num_workers': params.num_workers,
                                              'pin_memory': params.pin_memory
                                              },
                                 'inference': {'batch_size': params.batch_size_inference,
                                               'shuffle': False,
                                               'num_workers': params.num_workers,
                                               'pin_memory': params.pin_memory
                                               },
                                 },
               },

    'unet': {
        'architecture_config': {'model_params': {'n_filters': params.n_filters,
                                                 'conv_kernel': params.conv_kernel,
                                                 'pool_kernel': params.pool_kernel,
                                                 'pool_stride': params.pool_stride,
                                                 'repeat_blocks': params.repeat_blocks,
                                                 'batch_norm': params.use_batch_norm,
                                                 'dropout': params.dropout_conv,
                                                 'in_channels': params.image_channels,
                                                 'out_channels': params.channels_per_output,
                                                 'nr_outputs': params.nr_unet_outputs,
                                                 'encoder': params.encoder
                                                 },
                                'optimizer_params': {'lr': params.lr,
                                                     },
                                'regularizer_params': {'regularize': True,
                                                       'weight_decay_conv2d': params.l2_reg_conv,
                                                       },
                                'weights_init': {'function': 'he',
                                                 },
                                'loss_weights': {'bce_mask': params.bce_mask,
                                                 'dice_mask': params.dice_mask,
                                                 },
                                'weighted_cross_entropy': {'w0': params.w0,
                                                           'sigma': params.sigma,
                                                           'imsize': (params.image_h, params.image_w)},
                                'dice': {'smooth': params.dice_smooth,
                                         'dice_activation': params.dice_activation},
                                },
        'training_config': {'epochs': params.epochs_nr,
                            },
        'callbacks_config': {
            'model_checkpoint': {
                'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints', 'unet', 'best.torch'),
                'epoch_every': 1,
                'minimize': not params.validate_with_map
            },
            'exp_lr_scheduler': {'gamma': params.gamma,
                                 'epoch_every': 1},
            'plateau_lr_scheduler': {'lr_factor': params.lr_factor,
                                     'lr_patience': params.lr_patience,
                                     'epoch_every': 1},
            'training_monitor': {'batch_every': 1,
                                 'epoch_every': 1},
            'experiment_timing': {'batch_every': 10,
                                  'epoch_every': 1},
            'validation_monitor': {
                'epoch_every': 1,
                'data_dir': params.data_dir,
                'validate_with_map': params.validate_with_map,
                'small_annotations_size': params.small_annotations_size,
            },
            'neptune_monitor': {'model_name': 'unet',
                                'image_nr': 16,
                                'image_resize': 0.2,
                                'outputs_to_plot': params.unet_outputs_to_plot},
            'early_stopping': {'patience': params.patience,
                               'minimize': not params.validate_with_map},
        },
    },
    'tta_generator': {'flip_ud': True,
                      'flip_lr': True,
                      'rotation': True,
                      'color_shift_runs': False},
    'tta_aggregator': {'method': params.tta_aggregation_method,
                       'nthreads': params.num_threads
                       },
    'postprocessor': {'mask_dilation': {'dilate_selem_size': params.dilate_selem_size
                                        },
                      'mask_erosion': {'erode_selem_size': params.erode_selem_size
                                       },
                      'crf': {'apply_crf': params.apply_crf,
                              'nr_iter': params.nr_iter,
                              'compat_gaussian': params.compat_gaussian,
                              'sxy_gaussian': params.sxy_gaussian,
                              'compat_bilateral': params.compat_bilateral,
                              'sxy_bilateral': params.sxy_bilateral,
                              'srgb': params.srgb
                              },
                      'prediction_crop': {'h_crop': params.crop_image_h,
                                          'w_crop': params.crop_image_w
                                          },
                      'lightGBM': {'model_params': {'learning_rate': params.lgbm__learning_rate,
                                                    'boosting_type': 'gbdt',
                                                    'objective': 'binary',
                                                    'metric': 'binary_logloss',
                                                    'sub_feature': 0.5,
                                                    'num_leaves': params.lgbm__num_leaves,
                                                    'min_data': params.lgbm__min_data,
                                                    'max_depth': params.lgbm__max_depth},
                                   'training_params': {'number_boosting_rounds': params.lgbm__number_of_trees,
                                                       'early_stopping_rounds': params.lgbm__early_stopping},
                                   'train_size': params.lgbm__train_size,
                                   'target': params.lgbm__target
                                   },
                      'nms': {'iou_threshold': params.iou_threshold,
                              'n_threads': params.num_threads},
                      'feature_extractor': {'n_threads': params.num_threads}
                      }
})
