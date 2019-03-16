import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras_extensions import root_mean_square_error, theil_u, pearson_r, custom_epsilon_mean_absolute_percentage_error

from globals import features, encoded_features, labels, base_dir, number_of_gpus
from globals import global_batch_size, global_encode_features, global_time_steps, global_units\
    , global_model_type, global_epochs, global_wavelet_transform_iterations, global_dropout_rate, global_stateful, global_scale_type
from utils import create_data_directory, load_and_transform_data, save_model_with_additional_data, \
    load_additional_data, load_prediction_model, run_function_in_separate_process, plot_real_vs_predicted_data


def create_sequence_model(model_type, number_of_features, time_steps, batch_size, stateful, units, dropout_rate):
    """
    Creates the baseline model consisting of an either GRU or LSTM layer

    Arguments:
        model_type -- model to be used: gru or lstm
        number_of_features -- number of features provided as input
        time_steps -- value of the time_steps hyperparameter used, i.e. the number of time steps to look back
        batch_size -- value of the batch_size hyperparameter used
        stateful -- whether the LSTM model should be stateful or not, i.e. whether the state is preserverd betwenn batch iterations
        units -- value of the units hyperparameter used, i.e. the number of LSTM units in the LSTM layer
        dropout_rate -- dropout rate, i.e. fraction of the input units to drop

    Returns:

        model    -- Keras model
    """

    if stateful == False and batch_size != 1:
        batch_size = None

    with tf.device('/cpu:0'):

        # define model
        model = Sequential()
        if model_type == 'lstm':
            model.add(LSTM(units, activation='tanh', kernel_initializer='normal',
                           dropout=dropout_rate, recurrent_dropout=dropout_rate,
                           activity_regularizer=l2(0.000), recurrent_regularizer=l2(0.000),
                           input_shape=(time_steps, number_of_features),
                           batch_input_shape=(batch_size, time_steps, number_of_features),
                           stateful=stateful,
                           return_sequences=False))
        elif model_type == 'gru':
            model.add(GRU(units, activation='tanh', kernel_initializer='normal',
                          dropout=dropout_rate, recurrent_dropout=dropout_rate,
                          activity_regularizer=l2(0.000), recurrent_regularizer=l2(0.000),
                          input_shape=(time_steps, number_of_features),
                          batch_input_shape=(batch_size, time_steps, number_of_features),
                          stateful=stateful,
                          return_sequences=False))

        model.add(Dense(labels, kernel_initializer='normal', activation='sigmoid'))


    # GPU paralellization does not work for stateful and batch size 1
    if stateful == False and batch_size != 1:
        # Replicates the model on the number of given GPUs.
        # This assumes that the machine has that specified number of available GPUs.
        model = multi_gpu_model(model, gpus=number_of_gpus)

    optimizer = Adam(lr=0.05, decay=0.0005)

    model.compile(optimizer=optimizer, loss='mse', metrics=[root_mean_square_error, theil_u, pearson_r, custom_epsilon_mean_absolute_percentage_error])
    model.summary()

    return model

def Training(model,
             train_X, train_Y, dev_X, dev_Y,
             time_steps, batch_size, stateful, epochs, model_type, units, dropout_rate,
             scale_type, wavelet_transform_iterations, encode_features,
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
        model_type -- model to be used: lstm or gru
        units -- value of the units hyperparameter used, i.e. the number of LSTM units in the LSTM layer
        dropout_rate -- dropout rate, i.e. fraction of the input units to drop
        wavelet_transform_iterations -- value of wavelet_transform_iterations hyperparameter used, i.e. the number of times to apply wavelet transformation
        encode_features -- whether to encode the features or not

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
              shuffle=True)

    scores = model.evaluate(x=train_X, y=train_Y, batch_size=batch_size)
    rmse_train, theil_u_train, R_train, mape_train = scores[1], scores[2], scores[3], scores[4]

    scores = model.evaluate(x=dev_X, y=dev_Y, batch_size=batch_size)
    rmse_dev, theil_u_dev, R_dev, mape_dev = scores[1], scores[2], scores[3], scores[4]

    with open(output_directory + '/stats.txt', 'a') as f:
        f.write("%s: Train - %.6f %.6f %.6f %.6f Dev - %.6f %.6f %.6f %.6f Stateful: %s Time_steps: %d Batch size: %d Epochs: %d Model: %s Units: %d dropout rate: %.1f Scale type: %s Wavelet transform iterations: %d Encode features: %s\n" %
                (model.metrics_names[1], rmse_train, mape_train, theil_u_train, R_train, rmse_dev, mape_dev, theil_u_dev, R_dev,
                 str(stateful), time_steps, batch_size, epochs, model_type, units, dropout_rate, scale_type, wavelet_transform_iterations, str(encode_features)))


def SingleTraining(time_steps, batch_size, stateful, epochs, model_type, units, dropout_rate,
                   scale_type, wavelet_transform_iterations, encode_features,
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
        model_type -- model to be used: lstm or gru
        units -- value of the units hyperparameter used, i.e. the number of LSTM units in the LSTM layer
        dropout_rate -- dropout rate, fraction of the input units to drop
        scale_type -- type of scaling the data. Only Normalization and Standardization are supported
        wavelet_transform_iterations -- number of times to perform wavelet transformation on the input
        encode_features -- use encoded features or not

        output_directory -- directory where to save plot and statistics data.
        If save_configuration parameter is False the directory is created here using the local hyperparameter values
        save_configuration    -- flag to indicate that we are just training and we do not want to save the plots and the model.
        Used basically when searching for the hyperparameters outputing the best results

    Returns:

        model       -- Keras model
        scaler      -- used scaler
        output_directory -- where the model and data was written

    """

    adjust_to_multiple_of_batch_size = False

    if save_configuration == True:
        output_directory = create_data_directory(base_dir, time_steps=time_steps, batch_size=batch_size,
                                                 stateful=stateful,
                                                 epochs=epochs, model_type=model_type, rnn_units=units,
                                                 scale_type=scale_type,
                                                 wavelet_transform_iterations=wavelet_transform_iterations,
                                                 encode_features=encode_features)



    if stateful == True:
        adjust_to_multiple_of_batch_size = True

    # Based on the scale type apply on the training set the scaling and applying it after to dev and train sets
    train_x, train_y, scaler_X, scaler_Y = load_and_transform_data('data/train_X.csv', 'data/train_Y.csv', time_steps=time_steps, batch_size=batch_size,
                                                                   scale_type=scale_type, scaler_X=None, scaler_Y=None,
                                                                   wavelet_transform_iterations=wavelet_transform_iterations, encode_features=encode_features,
                                                                   adjust_to_multiple_of_batch_size=adjust_to_multiple_of_batch_size)

    dev_x, dev_y, _, _ = load_and_transform_data('data/dev_test_X.csv', 'data/dev_test_Y.csv', time_steps=time_steps, batch_size=batch_size,
                                                 scale_type=scale_type, scaler_X=scaler_X, scaler_Y=scaler_Y,
                                                 wavelet_transform_iterations=wavelet_transform_iterations, encode_features=encode_features,
                                                 adjust_to_multiple_of_batch_size=adjust_to_multiple_of_batch_size)

    model = None

    # when training we want to create and train the model in a separate process
    # so that the GPU memory is released between the runs, making the runs in the end independent
    if save_configuration == False:
        run_function_in_separate_process(CreateAndTrainModel, train_x, train_y, dev_x, dev_y,
                                         time_steps, batch_size, stateful,
                                         epochs, model_type, units, dropout_rate,
                                         scale_type,
                                         wavelet_transform_iterations,
                                         encode_features,
                                         output_directory,
                                         save_configuration)

    if save_configuration == True:

        model = CreateAndTrainModel(train_x, train_y, dev_x, dev_y,
                                    time_steps, batch_size, stateful,
                                    epochs, model_type, units, dropout_rate,
                                    scale_type,
                                    wavelet_transform_iterations,
                                    encode_features,
                                    output_directory,
                                    save_configuration)

        save_model_with_additional_data(model=model,
                                        scaler_X=scaler_X, scaler_Y=scaler_Y,
                                        output_directory=output_directory)

    return model, scaler_X, scaler_Y, output_directory


def CreateAndTrainModel(train_X, train_Y, dev_X, dev_Y,
                        time_steps, batch_size, stateful, epochs,
                        model_type, units, dropout_rate,
                        scale_type,
                        wavelet_transform_iterations,
                        encode_features,
                        output_directory, save_configuration):
    """

    function creating and training the model. Used mainly to be called in a different process

    Arguments:

        train_X     -- training input data
        train_Y     -- labels for the training input data
        dev_X       -- dev input data
        dev_Y       -- labels for the dev input data

        time_steps -- value of the time_steps hyperparameter used, i.e. the number of time steps to look back
        batch_size -- value of the batch_size hyperparameter used
        stateful   -- value of the stateful hyperparameter used
        epochs     -- value of the epcchs hyperparameter used, i.e. how many times should we go through our whole training set
        model_type -- model to be used: lstm or gru
        units -- value of the units hyperparameter used, i.e. the number of LSTM units in the LSTM layer
        dropout_rate -- dropout rate, fraction of the input units to drop
        scale_type -- type of scaling the data. Only Normalization and Standardization are supported
        wavelet_transform_iterations -- number of times to perform wavelet transformation on the input
        encode_features -- use encoded features or not

        output_directory -- directory where to save plot and statistics data.
        If save_configuration parameter is False the directory is created here using the local hyperparameter values
        save_configuration    -- flag to indicate that we are just training and we do not want to save the plots and the model.
        Used basically when searching for the hyperparameters outputing the best results

    Returns:

        model       -- Keras model when not called from a different process


    """

    # always generate the same random numbers
    np.random.seed(3)

    number_of_features = features
    if encode_features == True:
        number_of_features = encoded_features

    model = create_sequence_model(model_type=model_type, number_of_features=number_of_features,
                                  time_steps=time_steps, batch_size=batch_size, stateful=stateful,
                                  units=units, dropout_rate=dropout_rate)

    Training(model, train_X, train_Y, dev_X, dev_Y,
             time_steps=time_steps, batch_size=batch_size, stateful=stateful, epochs=epochs,
             model_type=model_type, units=units, dropout_rate=dropout_rate,
             scale_type=scale_type,
             wavelet_transform_iterations=wavelet_transform_iterations,
             encode_features=encode_features,
             output_directory=output_directory)

    #Due to the custom RMSE function when running the function in a separate process
    #the deserialization of the model does not find the custom function. Solve this later.
    #For now just return the model in case we don't run it in a separate process
    if save_configuration == True:
        return model

def SingleTrainingWithGlobalVariables():
    """

    function to train the model specified by the global parameters and save the configuration

    Arguments:

    Returns:

    """
    # always generate the same random numbers
    np.random.seed(3)

    SingleTraining(time_steps=global_time_steps, batch_size=global_batch_size, stateful=global_stateful,
                   epochs=global_epochs, model_type=global_model_type, units=global_units, dropout_rate=global_dropout_rate,
                   output_directory=base_dir,
                   scale_type=global_scale_type,
                   wavelet_transform_iterations=global_wavelet_transform_iterations,
                   encode_features=global_encode_features,
                   save_configuration=True)

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
    batch_size_list = [64]
    stateful_list = [False]
    epochs_list = [2000]
    model_types_list = ['gru', 'lstm']
    rnn_units_list = [128]
    dropout_rate_list = [0, 0.15]
    scale_type_list = ['normal']
    wavelet_transform_iterations_list = [0, 2]
    encode_features_list = [False, True]

    for time_steps in time_steps_list:
        for batch_size in batch_size_list:
            for stateful in stateful_list:
                for epochs in epochs_list:
                    for model_type in model_types_list:
                        for rnn_units in rnn_units_list:
                            for dropout_rate in dropout_rate_list:
                                for scale_type in scale_type_list:
                                    for wavelet_transform_iterations in wavelet_transform_iterations_list:
                                        for encode_features in encode_features_list:
                                            SingleTraining(time_steps=time_steps, batch_size=batch_size, stateful=stateful,
                                                           epochs=epochs, model_type=model_type, units=rnn_units, dropout_rate=dropout_rate,
                                                           output_directory=data_directory,
                                                           scale_type=scale_type,
                                                           wavelet_transform_iterations=wavelet_transform_iterations,
                                                           encode_features=encode_features,
                                                           save_configuration=False)


def SingleEvaluation(features_file, labels_file):
    """

    function used to evaluate the prediction model read from the models directory

    Arguments:

        features_file -- features to be used for predictions
        labels_file -- real values corresponding to the givel features

    Returns:
        y -- the real label values
        predictions -- the corresponding predicted values

    """
    scaler_X, scaler_Y = load_additional_data('models')

    adjust_to_multiple_of_batch_size = False

    if global_stateful == True:
        adjust_to_multiple_of_batch_size = True

    # Based on the scale type apply on the training set the scaling and applying it after to dev and train sets
    x, y, _, _ = load_and_transform_data(features_file, labels_file, time_steps=global_time_steps, batch_size=global_batch_size,
                                                                   scale_type=global_scale_type, scaler_X=scaler_X, scaler_Y=scaler_Y,
                                                                   wavelet_transform_iterations=global_wavelet_transform_iterations, encode_features=global_encode_features,
                                                                   adjust_to_multiple_of_batch_size=adjust_to_multiple_of_batch_size)

    model = load_prediction_model('models')

    predictions = model.predict(x=x, batch_size=global_batch_size)

    if scaler_Y is not None:
        y = scaler_Y.inverse_transform(y)
        predictions = scaler_Y.inverse_transform(predictions)

    return y, predictions


if __name__ == '__main__':

    GridSearchTraining()

