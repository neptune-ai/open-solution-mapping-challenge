from keras import regularizers
from keras.activations import relu
from keras.layers import Input, Embedding, PReLU, Bidirectional, Lambda, \
    CuDNNLSTM, CuDNNGRU, Conv1D, Dense, BatchNormalization, Dropout, SpatialDropout1D, \
    GlobalMaxPool1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.merge import add, concatenate
from keras.models import Model

from .contrib import AttentionWeightedAverage


def scnn(embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
         filter_nr, kernel_size, repeat_block, dense_size, repeat_dense, output_size, output_activation,
         max_pooling, mean_pooling, weighted_average_attention, concat_mode,
         dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
         conv_kernel_reg_l2, conv_bias_reg_l2,
         dense_kernel_reg_l2, dense_bias_reg_l2,
         use_prelu, use_batch_norm, batch_norm_first):
    input_text = Input(shape=(maxlen,))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=trainable_embedding)(
        input_text)

    x = dropout_block(dropout_embedding, dropout_mode)(x)

    for _ in range(repeat_block):
        x = convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, conv_dropout, dropout_mode,
                                conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first)(x)

    predictions = classification_block(dense_size=dense_size, repeat_dense=repeat_dense,
                                       output_size=output_size, output_activation=output_activation,
                                       max_pooling=max_pooling,
                                       mean_pooling=mean_pooling,
                                       weighted_average_attention=weighted_average_attention,
                                       concat_mode=concat_mode,
                                       dropout=dense_dropout,
                                       kernel_reg_l2=dense_kernel_reg_l2, bias_reg_l2=dense_bias_reg_l2,
                                       use_prelu=use_prelu, use_batch_norm=use_batch_norm,
                                       batch_norm_first=batch_norm_first)(x)
    model = Model(inputs=input_text, outputs=predictions)
    return model


def dpcnn(embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
          filter_nr, kernel_size, repeat_block, dense_size, repeat_dense, output_size, output_activation,
          max_pooling, mean_pooling, weighted_average_attention, concat_mode,
          dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
          conv_kernel_reg_l2, conv_bias_reg_l2,
          dense_kernel_reg_l2, dense_bias_reg_l2,
          use_prelu, use_batch_norm, batch_norm_first):
    """
    Note:
        Implementation of http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        post activation is used instead of pre-activation, could be worth exploring
    """

    input_text = Input(shape=(maxlen,))
    if embedding_matrix is not None:
        embedding = Embedding(max_features, embedding_size,
                              weights=[embedding_matrix], trainable=trainable_embedding)(input_text)
    else:
        embedding = Embedding(max_features, embedding_size)(input_text)

    embedding = dropout_block(dropout_embedding, dropout_mode)(embedding)

    x = convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, conv_dropout, dropout_mode,
                            conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first)(embedding)
    x = convolutional_block(filter_nr, kernel_size, conv_bias_reg_l2, use_prelu, conv_dropout, dropout_mode,
                            conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first)(x)
    if embedding_size == filter_nr:
        x = add([embedding, x])
    else:
        embedding_resized = shape_matching_layer(filter_nr, use_prelu, conv_kernel_reg_l2, conv_bias_reg_l2)(embedding)
        x = add([embedding_resized, x])
    for _ in range(repeat_block):
        x = dpcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, conv_dropout, dropout_mode,
                        conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first)(x)

    predictions = classification_block(dense_size=dense_size, repeat_dense=repeat_dense,
                                       output_size=output_size, output_activation=output_activation,
                                       max_pooling=max_pooling,
                                       mean_pooling=mean_pooling,
                                       weighted_average_attention=weighted_average_attention,
                                       concat_mode=concat_mode,
                                       dropout=dense_dropout,
                                       kernel_reg_l2=dense_kernel_reg_l2, bias_reg_l2=dense_bias_reg_l2,
                                       use_prelu=use_prelu, use_batch_norm=use_batch_norm,
                                       batch_norm_first=batch_norm_first)(x)
    model = Model(inputs=input_text, outputs=predictions)
    return model


def cudnn_lstm(embedding_matrix, embedding_size, trainable_embedding,
               maxlen, max_features,
               unit_nr, repeat_block,
               dense_size, repeat_dense, output_size, output_activation,
               max_pooling, mean_pooling, weighted_average_attention, concat_mode,
               dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
               rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
               dense_kernel_reg_l2, dense_bias_reg_l2,
               use_prelu, use_batch_norm, batch_norm_first):
    input_text = Input(shape=(maxlen,))
    if embedding_matrix is not None:
        x = Embedding(max_features,
                      embedding_size,
                      weights=[embedding_matrix],
                      trainable=trainable_embedding)(input_text)
    else:
        x = Embedding(max_features,
                      embedding_size)(input_text)

    x = dropout_block(dropout_embedding, dropout_mode)(x)

    for _ in range(repeat_block):
        x = cudnn_lstm_block(unit_nr=unit_nr, return_sequences=True, bidirectional=True,
                            kernel_reg_l2=rnn_kernel_reg_l2,
                            recurrent_reg_l2=rnn_recurrent_reg_l2,
                            bias_reg_l2=rnn_bias_reg_l2,
                            use_batch_norm=use_batch_norm, batch_norm_first=batch_norm_first,
                            dropout=rnn_dropout, dropout_mode=dropout_mode, use_prelu=use_prelu)(x)

    predictions = classification_block(dense_size=dense_size, repeat_dense=repeat_dense,
                                       output_size=output_size, output_activation=output_activation,
                                       max_pooling=max_pooling,
                                       mean_pooling=mean_pooling,
                                       weighted_average_attention=weighted_average_attention,
                                       concat_mode=concat_mode,
                                       dropout=dense_dropout,
                                       kernel_reg_l2=dense_kernel_reg_l2, bias_reg_l2=dense_bias_reg_l2,
                                       use_prelu=use_prelu, use_batch_norm=use_batch_norm,
                                       batch_norm_first=batch_norm_first)(x)
    model = Model(inputs=input_text, outputs=predictions)
    return model


def cudnn_gru(embedding_matrix, embedding_size, trainable_embedding,
              maxlen, max_features,
              unit_nr, repeat_block,
              dense_size, repeat_dense, output_size, output_activation,
              max_pooling, mean_pooling, weighted_average_attention, concat_mode,
              dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
              rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
              dense_kernel_reg_l2, dense_bias_reg_l2,
              use_prelu, use_batch_norm, batch_norm_first):
    input_text = Input(shape=(maxlen,))
    if embedding_matrix is not None:
        x = Embedding(max_features,
                      embedding_size,
                      weights=[embedding_matrix],
                      trainable=trainable_embedding)(input_text)
    else:
        x = Embedding(max_features,
                      embedding_size)(input_text)

    x = dropout_block(dropout_embedding, dropout_mode)(x)

    for _ in range(repeat_block):
        x = cudnn_gru_block(unit_nr=unit_nr, return_sequences=True, bidirectional=True,
                            kernel_reg_l2=rnn_kernel_reg_l2,
                            recurrent_reg_l2=rnn_recurrent_reg_l2,
                            bias_reg_l2=rnn_bias_reg_l2,
                            use_batch_norm=use_batch_norm, batch_norm_first=batch_norm_first,
                            dropout=rnn_dropout, dropout_mode=dropout_mode, use_prelu=use_prelu)(x)

    predictions = classification_block(dense_size=dense_size, repeat_dense=repeat_dense,
                                       output_size=output_size, output_activation=output_activation,
                                       max_pooling=max_pooling,
                                       mean_pooling=mean_pooling,
                                       weighted_average_attention=weighted_average_attention,
                                       concat_mode=concat_mode,
                                       dropout=dense_dropout,
                                       kernel_reg_l2=dense_kernel_reg_l2, bias_reg_l2=dense_bias_reg_l2,
                                       use_prelu=use_prelu, use_batch_norm=use_batch_norm,
                                       batch_norm_first=batch_norm_first)(x)
    model = Model(inputs=input_text, outputs=predictions)
    return model


def vdcnn(embedding_size, maxlen, max_features,
          filter_nr, kernel_size, repeat_block, dense_size, repeat_dense, output_size, output_activation,
          max_pooling, mean_pooling, weighted_average_attention, concat_mode,
          dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
          conv_kernel_reg_l2, conv_bias_reg_l2,
          dense_kernel_reg_l2, dense_bias_reg_l2,
          use_prelu, use_batch_norm, batch_norm_first):
    """
    Note:
        Implementation of http://www.aclweb.org/anthology/E17-1104
        We didn't use k-max pooling but GlobalMaxPool1D at the end and didn't explore it in the
        intermediate layers.
    """

    input_text = Input(shape=(maxlen,))
    x = Embedding(input_dim=max_features, output_dim=embedding_size)(input_text)

    x = dropout_block(dropout_embedding, dropout_mode)(x)

    x = convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, conv_dropout, dropout_mode,
                            conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first)(x)

    for i in range(repeat_block):
        if i + 1 != repeat_block:
            x = vdcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, conv_dropout, dropout_mode,
                            conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first, last_block=False)(x)
        else:
            x = vdcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, conv_dropout, dropout_mode,
                            conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first, last_block=True)(x)

    predictions = classification_block(dense_size=dense_size, repeat_dense=repeat_dense,
                                       output_size=output_size, output_activation=output_activation,
                                       max_pooling=max_pooling,
                                       mean_pooling=mean_pooling,
                                       weighted_average_attention=weighted_average_attention,
                                       concat_mode=concat_mode,
                                       dropout=dense_dropout,
                                       kernel_reg_l2=dense_kernel_reg_l2, bias_reg_l2=dense_bias_reg_l2,
                                       use_prelu=use_prelu, use_batch_norm=use_batch_norm,
                                       batch_norm_first=batch_norm_first)(x)
    model = Model(inputs=input_text, outputs=predictions)
    return model


def classification_block(dense_size, repeat_dense, output_size, output_activation,
                         max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                         dropout,
                         kernel_reg_l2, bias_reg_l2,
                         use_prelu, use_batch_norm, batch_norm_first):
    def f(x):
        if max_pooling:
            x_max = GlobalMaxPool1D()(x)
        else:
            x_max = None

        if mean_pooling:
            x_mean = GlobalAveragePooling1D()(x)
        else:
            x_mean = None
        if weighted_average_attention:
            x_att = AttentionWeightedAverage()(x)
        else:
            x_att = None

        x = [xi for xi in [x_max, x_mean, x_att] if xi is not None]
        if len(x) == 1:
            x = x[0]
        else:
            if concat_mode == 'concat':
                x = concatenate(x, axis=-1)
            else:
                NotImplementedError('only mode concat for now')

        for _ in range(repeat_dense):
            x = dense_block(dense_size=dense_size,
                            use_batch_norm=use_batch_norm,
                            use_prelu=use_prelu,
                            dropout=dropout,
                            kernel_reg_l2=kernel_reg_l2,
                            bias_reg_l2=bias_reg_l2,
                            batch_norm_first=batch_norm_first)(x)

        x = Dense(output_size, activation=output_activation)(x)
        return x

    return f


def dropout_block(dropout, dropout_mode):
    def f(x):
        if dropout_mode == 'spatial':
            x = SpatialDropout1D(dropout)(x)
        elif dropout_mode == 'simple':
            x = Dropout(dropout)(x)
        else:
            raise NotImplementedError('spatial and simple modes are supported')
        return x

    return f


def prelu_block(use_prelu):
    def f(x):
        if use_prelu:
            x = PReLU()(x)
        else:
            x = Lambda(relu)(x)
        return x

    return f


def bn_relu_dropout_block(use_batch_norm, use_prelu, dropout, dropout_mode, batch_norm_first):
    def f(x):
        if use_batch_norm and batch_norm_first:
            x = BatchNormalization()(x)

        x = prelu_block(use_prelu)(x)
        x = dropout_block(dropout, dropout_mode)(x)

        if use_batch_norm and not batch_norm_first:
            x = BatchNormalization()(x)
        return x

    return f


def convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                        kernel_reg_l2, bias_reg_l2, batch_norm_first):
    def f(x):
        x = Conv1D(filter_nr, kernel_size=kernel_size, padding='same', activation='linear',
                   kernel_regularizer=regularizers.l2(kernel_reg_l2),
                   bias_regularizer=regularizers.l2(bias_reg_l2))(x)
        x = bn_relu_dropout_block(use_batch_norm=use_batch_norm,
                                  batch_norm_first=batch_norm_first,
                                  dropout=dropout,
                                  dropout_mode=dropout_mode,
                                  use_prelu=use_prelu)(x)
        return x

    return f


def shape_matching_layer(filter_nr, use_prelu, kernel_reg_l2, bias_reg_l2):
    def f(x):
        x = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',
                   kernel_regularizer=regularizers.l2(kernel_reg_l2),
                   bias_regularizer=regularizers.l2(bias_reg_l2))(x)
        x = prelu_block(use_prelu)(x)
        return x

    return f


def cudnn_lstm_block(unit_nr, return_sequences, bidirectional,
                     kernel_reg_l2, recurrent_reg_l2, bias_reg_l2,
                     use_batch_norm, batch_norm_first,
                     dropout, dropout_mode, use_prelu):
    def f(x):
        gru_layer = CuDNNLSTM(uunits=unit_nr, return_sequences=return_sequences,
                              kernel_regularizer=regularizers.l2(kernel_reg_l2),
                              recurrent_regularizer=regularizers.l2(recurrent_reg_l2),
                              bias_regularizer=regularizers.l2(bias_reg_l2)
                              )
        if bidirectional:
            x = Bidirectional(gru_layer)(x)
        else:
            x = gru_layer(x)
        x = bn_relu_dropout_block(use_batch_norm=use_batch_norm, batch_norm_first=batch_norm_first,
                                  dropout=dropout, dropout_mode=dropout_mode,
                                  use_prelu=use_prelu)(x)
        return x

    return f


def cudnn_gru_block(unit_nr, return_sequences, bidirectional,
                    kernel_reg_l2, recurrent_reg_l2, bias_reg_l2,
                    use_batch_norm, batch_norm_first,
                    dropout, dropout_mode, use_prelu):
    def f(x):
        gru_layer = CuDNNGRU(units=unit_nr, return_sequences=return_sequences,
                             kernel_regularizer=regularizers.l2(kernel_reg_l2),
                             recurrent_regularizer=regularizers.l2(recurrent_reg_l2),
                             bias_regularizer=regularizers.l2(bias_reg_l2)
                             )
        if bidirectional:
            x = Bidirectional(gru_layer)(x)
        else:
            x = gru_layer(x)
        x = bn_relu_dropout_block(use_batch_norm=use_batch_norm, batch_norm_first=batch_norm_first,
                                  dropout=dropout, dropout_mode=dropout_mode,
                                  use_prelu=use_prelu)(x)
        return x

    return f


def dense_block(dense_size, use_batch_norm, use_prelu, dropout, kernel_reg_l2, bias_reg_l2,
                batch_norm_first):
    def f(x):
        x = Dense(dense_size, activation='linear',
                  kernel_regularizer=regularizers.l2(kernel_reg_l2),
                  bias_regularizer=regularizers.l2(bias_reg_l2))(x)

        x = bn_relu_dropout_block(use_batch_norm=use_batch_norm,
                                  use_prelu=use_prelu,
                                  dropout=dropout,
                                  dropout_mode='simple',
                                  batch_norm_first=batch_norm_first)(x)
        return x

    return f


def dpcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                kernel_reg_l2, bias_reg_l2, batch_norm_first):
    def f(x):
        x = MaxPooling1D(pool_size=3, strides=2)(x)
        main = convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                                   kernel_reg_l2, bias_reg_l2, batch_norm_first)(x)
        main = convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                                   kernel_reg_l2, bias_reg_l2, batch_norm_first)(main)
        x = add([main, x])
        return x

    return f


def vdcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                kernel_reg_l2, bias_reg_l2, batch_norm_first, last_block):
    def f(x):
        main = convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                                   kernel_reg_l2, bias_reg_l2, batch_norm_first)(x)
        x = add([main, x])
        main = convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                                   kernel_reg_l2, bias_reg_l2, batch_norm_first)(x)
        x = add([main, x])
        if not last_block:
            x = MaxPooling1D(pool_size=3, strides=2)(x)
        return x

    return f
