import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params

ctx = neptune.Context()
params = read_params(ctx)

SIZE_COLUMNS = ['height', 'width']
X_COLUMNS = ['file_path_image']
Y_COLUMNS = ['file_path_mask_eroded_30']
Y_COLUMNS_SCORING = ['ImageId']
CATEGORY_IDS = [None, 100]
MEAN = [0., 0., 0.]
STD = [1., 1., 1.]

GLOBAL_CONFIG = {'exp_root': params.experiment_dir,
                 'load_in_memory': params.load_in_memory,
                 'num_workers': params.num_workers,
                 'num_classes': 2,
                 'img_H-W': (params.image_h, params.image_w),
                 'batch_size_train': params.batch_size_train,
                 'batch_size_inference': params.batch_size_inference,
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
    'loader': {'dataset_params': {'h': params.image_h,
                                  'w': params.image_w
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
                                                 'mask': params.mask,
                                                 },
                                },
        'training_config': {'epochs': params.epochs_nr,
                            'loss_function': {'w0': params.w0,
                                              'sigma': params.sigma},
                            },
        'callbacks_config': {
            'model_checkpoint': {
                'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints', 'unet', 'best.torch'),
                'epoch_every': 1},
            'exp_lr_scheduler': {'gamma': params.gamma,
                                 'epoch_every': 1},
            'plateau_lr_scheduler': {'lr_factor': params.lr_factor,
                                     'lr_patience': params.lr_patience,
                                     'epoch_every': 1},
            'training_monitor': {'batch_every': 0,
                                 'epoch_every': 1},
            'experiment_timing': {'batch_every': 0,
                                  'epoch_every': 1},
            'validation_monitor': {'epoch_every': 1},
            'neptune_monitor': {'model_name': 'unet',
                                'image_nr': 4,
                                'image_resize': 0.2,
                                'outputs_to_plot': params.unet_outputs_to_plot},
            'early_stopping': {'patience': params.patience},
        },
    },
    'dropper': {'min_size': params.min_nuclei_size},
    'postprocessor': {'erode_selem_size': params.erode_selem_size,
                      'dilate_selem_size': params.dilate_selem_size,
                      'crf': {'apply_crf': params.apply_crf,
                              'nr_iter': params.nr_iter,
                              'compat_gaussian': params.compat_gaussian,
                              'sxy_gaussian': params.sxy_gaussian,
                              'compat_bilateral': params.compat_bilateral,
                              'sxy_bilateral': params.sxy_bilateral,
                              'srgb': params.srgb
                              }}
})
