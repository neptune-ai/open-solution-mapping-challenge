import shutil

from keras.models import load_model

from ..base import BaseTransformer
from .contrib import AttentionWeightedAverage
from .architectures import vdcnn, scnn, dpcnn, cudnn_gru, cudnn_lstm


class KerasModelTransformer(BaseTransformer):
    """
    Todo:
        load the best model at the end of the fit and save it
    """

    def __init__(self, architecture_config, training_config, callbacks_config):
        self.architecture_config = architecture_config
        self.training_config = training_config
        self.callbacks_config = callbacks_config

    def reset(self):
        self.model = self._build_model(**self.architecture_config)

    def _compile_model(self, model_params, optimizer_params):
        model = self._build_model(**model_params)
        optimizer = self._build_optimizer(**optimizer_params)
        loss = self._build_loss()
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def _create_callbacks(self, **kwargs):
        return NotImplementedError

    def _build_model(self, **kwargs):
        return NotImplementedError

    def _build_optimizer(self, **kwargs):
        return NotImplementedError

    def _build_loss(self, **kwargs):
        return NotImplementedError

    def save(self, filepath):
        checkpoint_callback = self.callbacks_config.get('model_checkpoint')
        if checkpoint_callback:
            checkpoint_filepath = checkpoint_callback['filepath']
            shutil.copyfile(checkpoint_filepath, filepath)
        else:
            self.model.save(filepath)

    def load(self, filepath):
        self.model = load_model(filepath,
                                custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
        return self


class ClassifierXY(KerasModelTransformer):
    def fit(self, X, y, validation_data, *args, **kwargs):
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.model = self._compile_model(**self.architecture_config)

        self.model.fit(X, y,
                       validation_data=validation_data,
                       callbacks=self.callbacks,
                       verbose=1,
                       **self.training_config)
        return self

    def transform(self, X, y=None, validation_data=None, *args, **kwargs):
        predictions = self.model.predict(X, verbose=1)
        return {'prediction_probability': predictions}


class ClassifierGenerator(KerasModelTransformer):
    def fit(self, datagen, validation_datagen, *args, **kwargs):
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.model = self._compile_model(**self.architecture_config)

        train_flow, train_steps = datagen
        valid_flow, valid_steps = validation_datagen
        self.model.fit_generator(train_flow,
                                 steps_per_epoch=train_steps,
                                 validation_data=valid_flow,
                                 validation_steps=valid_steps,
                                 callbacks=self.callbacks,
                                 verbose=1,
                                 **self.training_config)
        return self

    def transform(self, datagen, validation_datagen=None, *args, **kwargs):
        test_flow, test_steps = datagen
        predictions = self.model.predict_generator(test_flow, test_steps, verbose=1)
        return {'prediction_probability': predictions}


class PretrainedEmbeddingModel(ClassifierXY):
    def fit(self, X, y, validation_data, embedding_matrix):
        X_valid, y_valid = validation_data
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.architecture_config['model_params']['embedding_matrix'] = embedding_matrix
        self.model = self._compile_model(**self.architecture_config)
        self.model.fit(X, y,
                       validation_data=[X_valid, y_valid],
                       callbacks=self.callbacks,
                       verbose=1,
                       **self.training_config)
        return self

    def transform(self, X, y=None, validation_data=None, embedding_matrix=None):
        predictions = self.model.predict(X, verbose=1)
        return {'prediction_probability': predictions}


class CharVDCNNTransformer(ClassifierXY):
    def _build_model(self, embedding_size, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        return vdcnn(embedding_size, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first)


class WordSCNNTransformer(PretrainedEmbeddingModel):
    def _build_model(self, embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        return scnn(embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
                    filter_nr, kernel_size, repeat_block,
                    dense_size, repeat_dense, output_size, output_activation,
                    max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                    dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                    conv_kernel_reg_l2, conv_bias_reg_l2,
                    dense_kernel_reg_l2, dense_bias_reg_l2,
                    use_prelu, use_batch_norm, batch_norm_first)


class WordDPCNNTransformer(PretrainedEmbeddingModel):
    def _build_model(self, embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        """
        Implementation of http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        """
        return dpcnn(embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first)


class WordCuDNNLSTMTransformer(PretrainedEmbeddingModel):
    def _build_model(self, embedding_matrix, embedding_size, trainable_embedding,
                     maxlen, max_features,
                     unit_nr, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
                     rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        return cudnn_lstm(embedding_matrix, embedding_size, trainable_embedding,
                          maxlen, max_features,
                          unit_nr, repeat_block,
                          dense_size, repeat_dense, output_size, output_activation,
                          max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                          dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
                          rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
                          dense_kernel_reg_l2, dense_bias_reg_l2,
                          use_prelu, use_batch_norm, batch_norm_first)


class WordCuDNNGRUTransformer(PretrainedEmbeddingModel):
    def _build_model(self, embedding_matrix, embedding_size, trainable_embedding,
                     maxlen, max_features,
                     unit_nr, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
                     rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        return cudnn_gru(embedding_matrix, embedding_size, trainable_embedding,
                         maxlen, max_features,
                         unit_nr, repeat_block,
                         dense_size, repeat_dense, output_size, output_activation,
                         max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                         dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
                         rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
                         dense_kernel_reg_l2, dense_bias_reg_l2,
                         use_prelu, use_batch_norm, batch_norm_first)
