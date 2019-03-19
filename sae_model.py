import tensorflow as tf
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import multi_gpu_model
from keras_extensions import root_mean_square_error

from globals import features, encoded_features, labels, base_dir, number_of_gpus
from utils import create_data_directory, load_and_transform_data, save_model_with_additional_data, run_function_in_separate_process

def create_stacked_auto_encoder_model(time_steps, dropout_rate, l2_reg_rate, level1_units, level2_units):
    """


    Arguments:


    Returns:


    """

    with tf.device('/cpu:0'):

        input_layer = Input(shape=(time_steps, features), batch_shape=(None, time_steps, features))

        # encoder 1
        encoding_layer_1 = Dense(level1_units, activation='relu', activity_regularizer=regularizers.l2(l2_reg_rate),
                                 kernel_initializer='glorot_normal')(input_layer)
        encoding_layer_1 = Dropout(dropout_rate)(encoding_layer_1)

        # encoder 2
        encoding_layer_2 = Dense(level2_units, activation='relu', activity_regularizer=regularizers.l2(l2_reg_rate),
                                 kernel_initializer='glorot_normal')(encoding_layer_1)
        encoding_layer_2 = Dropout(dropout_rate)(encoding_layer_2)

        # encoder 3
        encoding_layer_3 = Dense(encoded_features, activation='relu', activity_regularizer=regularizers.l2(l2_reg_rate),
                                 kernel_initializer='glorot_normal')(encoding_layer_2)
        encoding_layer_3 = Dropout(dropout_rate)(encoding_layer_3)

        # decoder 1
        decoding_layer_1 = Dense(level2_units, activation='relu', activity_regularizer=regularizers.l2(l2_reg_rate),
                                 kernel_initializer='glorot_normal')(encoding_layer_3)
        decoding_layer_1 = Dropout(dropout_rate)(decoding_layer_1)

        # decoder 2
        decoding_layer_2 = Dense(level1_units, activation='relu', activity_regularizer=regularizers.l2(l2_reg_rate),
                                 kernel_initializer='glorot_normal')(decoding_layer_1)
        decoding_layer_2 = Dropout(dropout_rate)(decoding_layer_2)

        # decoded
        decoded = Dense(features, activation='sigmoid', kernel_initializer='normal')(decoding_layer_2)

        # define model
        model = Model(inputs=input_layer, outputs=decoded)
        encoding_model = Model(inputs=input_layer, outputs=encoding_layer_3)


    # GPU paralellization does not work for stateful and batch size 1

    # Replicates the model on the number of given GPUs.
    # This assumes that the machine has that specified number of available GPUs.
    model = multi_gpu_model(model, gpus=number_of_gpus)

    optimizer = Adam()

    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    return model, encoding_model

def Training(model,
             train_X, train_Y, dev_X, dev_Y,
             time_steps, batch_size, epochs, dropout_rate, l2_reg_rate, level1_units, level2_units,
             scale_type, wavelet_transform_iterations,
             output_directory):
    """
    Trains the model, evaluates it separately on the train and dev set to get
    the rmse statistic for both sets which are printed in an output file.
    If called from a standalone context it saves the model with the weights and scaler used

    Arguments:
        model       -- Keras model
        train_X     -- training input data
        train_Y     -- labels for the training input data
        dev_X       -- dev input data
        dev_Y       -- labels for the dev input data

        time_steps -- value of the time_steps hyperparameter used, i.e. the number of time steps to look back
        batch_size -- value of the batch_size hyperparameter used
        stateful -- whether the LSTM model should be stateful or not, i.e. whether the state is preserverd betwenn batch iterations
        epochs     -- value of the epcchs hyperparameter used, i.e. how many times should we go through our whole training set
        lstm_units -- value of the lstm_units hyperparameter used, i.e. the number of LSTM units in the LSTM layer
        dropout_rate -- dropout rate, i.e. fraction of the input units to drop
        wavelet_transform_iterations -- value of wavelet_transform_iterations hyperparameter used, i.e. the number of times to apply wavelet transformation

        output_directory -- directory where to save plot and statistics data

    Returns:

    """

    # fit model
    model.fit(x=train_X,
              y=train_Y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(dev_X, dev_Y),
              shuffle=False)

    scores = model.evaluate(x=train_X, y=train_Y, batch_size=batch_size)
    rmse_train = scores

    scores = model.evaluate(x=dev_X, y=dev_Y, batch_size=batch_size)
    rmse_dev = scores

    with open(output_directory + '/stats.txt', 'a') as f:
        f.write("%s: Train - %.6f  Dev - %.6f Time_steps: %d Batch size: %d Epochs: %d Level 1 units: %d Level 2 units: %d Dropout rate: %.2f l2_reg_rate: %.3f Scale type: %s Wavelet transform iterations: %d\n" %
                (model.metrics_names[0], rmse_train, rmse_dev, time_steps, batch_size, epochs, level1_units, level2_units, dropout_rate, l2_reg_rate, scale_type, wavelet_transform_iterations))


def SingleTraining(time_steps, batch_size, epochs, dropout_rate, l2_reg_rate, level1_units, level2_units,
                   scale_type, wavelet_transform_iterations, include_tweets_sentiment,
                   output_directory, save_configuration):
    """
    Wrapper function over Training that previously loads the training and dev sets
    creates the output directory based on hyperparameters value if requested and
    calls the Training function

    Arguments:

        time_steps -- value of the time_steps hyperparameter used, i.e. the number of time steps to look back
        batch_size -- value of the batch_size hyperparameter used
        stateful   -- value of the stateful hyperparameter used
        epochs     -- value of the epcchs hyperparameter used, i.e. how many times should we go through our whole training set
        lstm_units -- value of the lstm_units hyperparameter used, i.e. the number of LSTM units in the LSTM layer
        dropout_rate -- dropout rate, fraction of the input units to drop

        output_directory -- directory where to save plot and statistics data.
        If save_configuration parameter is False the directory is created here using the local hyperparameter values
        scale_type -- type of scaling the data. Only Normalization and Standardization are supported
        wavelet_transform_iterations -- number of times to perform wavelet transformation on the input
        include_tweets_sentiment -- include the tweets sentiment as feature
        save_configuration    -- flag to indicate that we are just training and we do not want to save the plots and the model.
        Used basically when searching for the hyperparameters outputing the best results

    Returns:

        model       -- Keras model
        scaler      -- used scaler
        output_directory -- where the model and data was written

    """

    normalize_input = False
    standardize_input = False
    adjust_to_multiple_of_batch_size = False

    if save_configuration == True:
        output_directory = create_data_directory(base_dir + '/sae', time_steps=time_steps, batch_size=batch_size,
                                                 epochs=epochs, level1_units=level1_units, level2_units=level2_units,
                                                 scale_type=scale_type,
                                                 wavelet_transform_iterations=wavelet_transform_iterations,
                                                 include_tweets_sentiment=include_tweets_sentiment)

    # Based on the scale type apply on the training set the scaling and applying it after to dev and train sets
    train_x, train_y, scaler_X, scaler_Y = load_and_transform_data('data/train_X.csv', 'data/train_Y.csv', time_steps=time_steps, batch_size=batch_size,
                                                                   scale_type=scale_type, scaler_X=None, scaler_Y=None,
                                                                   wavelet_transform_iterations=wavelet_transform_iterations,
                                                                   encode_features=False,
                                                                   include_tweets_sentiment=include_tweets_sentiment,
                                                                   adjust_to_multiple_of_batch_size=adjust_to_multiple_of_batch_size)

    dev_x, dev_y, _, _ = load_and_transform_data('data/dev_test_X.csv', 'data/dev_test_Y.csv', time_steps=time_steps, batch_size=batch_size,
                                                 scale_type=scale_type, scaler_X=scaler_X, scaler_Y=scaler_Y,
                                                 wavelet_transform_iterations=wavelet_transform_iterations,
                                                 encode_features=False,
                                                 include_tweets_sentiment=include_tweets_sentiment,
                                                 adjust_to_multiple_of_batch_size=adjust_to_multiple_of_batch_size)

    # when training we want to create and train the model in a separate process
    # so that the GPU memory is released between the runs, making the runs in the end independent
    # we want the output to be the same as the input
    run_function_in_separate_process(CreateAndTrainModel, train_x, train_x, dev_x, dev_x,
                                     time_steps, batch_size,
                                     epochs, dropout_rate, l2_reg_rate, level1_units, level2_units,
                                     scale_type,
                                     wavelet_transform_iterations,
                                     output_directory,
                                     save_configuration)


def CreateAndTrainModel(train_X, train_Y, dev_X, dev_Y,
                        time_steps, batch_size,
                        epochs, dropout_rate, l2_reg_rate, level1_units, level2_units,
                        scale_type,
                        wavelet_transform_iterations,
                        output_directory,
                        save_configuration):
    # always generate the same random numbers
    np.random.seed(3)

    model, encoding_model = create_stacked_auto_encoder_model(time_steps=time_steps,
                                                              dropout_rate=dropout_rate, l2_reg_rate=l2_reg_rate,
                                                              level1_units=level1_units, level2_units=level2_units)

    Training(model, train_X, train_Y, dev_X, dev_Y,
             time_steps=time_steps, batch_size=batch_size, epochs=epochs,
             dropout_rate=dropout_rate, l2_reg_rate=l2_reg_rate, level1_units=level1_units, level2_units=level2_units,
             scale_type=scale_type,
             wavelet_transform_iterations=wavelet_transform_iterations,
             output_directory=output_directory)

    if save_configuration == True:
        # save the encoder model
        save_model_with_additional_data(model=encoding_model,
                                        scaler_X=None, scaler_Y=None,
                                        output_directory=output_directory)

def GridSearchTraining():
    """
    Wrapper function over SingleTraining that is a manual implementation of Grid Search
    Each Hyperparameter is actually an array and the model is trained on each combination.
    The statistic rmse is saved in a file for each run so that we can see the best results

    Arguments:

    Returns:

    """
    # always generate the same random numbers
    np.random.seed(3)

    data_directory = create_data_directory(base_dir)

    time_steps_list = [1]
    batch_size_list = [128]
    epochs_list = [5000]
    dropout_rate_list = [0.2]
    l2_reg_rate_list = [0.0]
    level1_units_list = [15]
    level2_units_list = [12]
    scale_type_list = ['normal']
    wavelet_transform_iterations_list = [1]
    include_tweets_sentiment_list = [True]

    for time_steps in time_steps_list:
        for batch_size in batch_size_list:
                for epochs in epochs_list:
                    for dropout_rate in dropout_rate_list:
                        for l2_reg_rate in l2_reg_rate_list:
                            for level1_units in level1_units_list:
                                for level2_units in level2_units_list:
                                    for scale_type in scale_type_list:
                                        for wavelet_transform_iterations in wavelet_transform_iterations_list:
                                            for include_tweets_sentiment in include_tweets_sentiment_list:
                                                SingleTraining(time_steps=time_steps, batch_size=batch_size,
                                                               epochs=epochs, dropout_rate=dropout_rate, l2_reg_rate=l2_reg_rate,
                                                               level1_units=level1_units, level2_units=level2_units,
                                                               output_directory=data_directory,
                                                               scale_type=scale_type,
                                                               wavelet_transform_iterations=wavelet_transform_iterations,
                                                               include_tweets_sentiment=include_tweets_sentiment,
                                                               save_configuration=True)

if __name__ == '__main__':

    GridSearchTraining()