#
# The SELDnet architecture
#
import tensorflow as tf
# from tensorflow import keras
# from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate
# from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
# from keras.layers.recurrent import GRU
# # from keras.layers.normalization import BatchNormalization
# from keras.models import Model
# from keras.layers.wrappers import TimeDistributed
# # from keras.optimizers import Adam
# # from keras.models import load_model
# import keras
# tf.keras.backend.set_image_data_format('channels_first')
# from IPython import embed
# import numpy as np
import parameter
import cls_data_generator
# import cls_feature_class


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, weights, doa_objective, is_accdoa):
    # model definition
    spec_start = tf.keras.Input(shape=(data_in[-3], data_in[-2], data_in[-1]))

    # CNN
    spec_cnn = spec_start
    for i, convCnt in enumerate(f_pool_size):
        spec_cnn = tf.keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
        spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
        spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
        spec_cnn = tf.keras.layers.Dropout(dropout_rate)(spec_cnn)
    spec_cnn = tf.keras.layers.Permute((2, 1, 3))(spec_cnn)

    # RNN    
    spec_rnn = tf.keras.layers.Reshape((data_out[-2] if is_accdoa else data_out[0][-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)

    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(nb_fnn_filt))(doa)
        doa = tf.keras.layers.Dropout(dropout_rate)(doa)

    doa = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(data_out[-1] if is_accdoa else data_out[1][-1]))(doa)
    doa = tf.keras.layers.Activation('tanh', name='doa_out')(doa)

    model = None
    if is_accdoa:
        model = tf.keras.Model(inputs=spec_start, outputs=doa)
        model.compile(optimizer='Adam', loss='mse')
    else:
        # FC - SED
        sed = spec_rnn
        for nb_fnn_filt in fnn_size:
            sed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(nb_fnn_filt))(sed)
            sed = tf.keras.layers.Dropout(dropout_rate)(sed)
        sed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(data_out[0][-1]))(sed)
        sed = tf.keras.layers.Activation('sigmoid', name='sed_out')(sed)

        if doa_objective == 'mse':
            model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
            model.compile(optimizer='Adam', loss=['binary_crossentropy', 'mse'], loss_weights=weights)
        elif doa_objective == 'masked_mse':
            doa_concat = tf.keras.layers.Concatenate(axis=-1, name='doa_concat')([sed, doa])
            model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa_concat])
            model.compile(optimizer='Adam', loss=['binary_crossentropy', masked_mse], loss_weights=weights)
        else:
            print('ERROR: Unknown doa_objective: {}'.format(doa_objective))
            exit()
    model.summary()
    return model


def masked_mse(y_gt, model_out):
    nb_classes = 12 #TODO fix this hardcoded value of number of classes
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :nb_classes] >= 0.5 
    sed_out = tf.keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = tf.keras.backend.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights 
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_gt[:, :, nb_classes:] - model_out[:, :, nb_classes:]) * sed_out))/tf.keras.backend.sum(sed_out)


def load_seld_model(model_file, doa_objective):
    if doa_objective == 'mse':
        return tf.keras.load_model(model_file)
    elif doa_objective == 'masked_mse':
        return tf.keras.load_model(model_file, custom_objects={'masked_mse': masked_mse})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()


if __name__ == "__main__":
    task_id = 1
    params = parameter.get_params()
    if params['mode'] == 'dev':
        test_splits = [6]
        val_splits = [5]
        train_splits = [[1, 2, 3, 4]]

    elif params['mode'] == 'eval':
        test_splits = [[7, 8]]
        val_splits = [[6]]
        train_splits = [[1, 2, 3, 4, 5]]
    # split_cnt, split = enumerate(test_splits)
    data_gen_train = cls_data_generator.DataGenerator(
        params=params, split=[1, 2, 3, 4]
    )
    data_in, data_out = data_gen_train.get_data_sizes()
    model = get_model(data_in=data_in,
                      data_out=data_out,
                      dropout_rate=params['dropout_rate'],
                      nb_cnn2d_filt=params['nb_cnn2d_filt'],
                      f_pool_size=params['f_pool_size'],
                      t_pool_size=params['t_pool_size'],
                      rnn_size=params['rnn_size'],
                      fnn_size=params['fnn_size'],
                      weights=params['loss_weights'],
                      doa_objective=params['doa_objective'],
                      is_accdoa=params['is_accdoa'])
    print("I wish this was written good")