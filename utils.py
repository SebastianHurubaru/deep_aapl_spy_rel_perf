import time
import datetime
import os
import traceback
import sys

import numpy as np
import pandas as pa
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
from multiprocessing import Process, Queue

import pywt

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.backend import clear_session

from keras_extensions import root_mean_square_error, theil_u, pearson_r, custom_epsilon_mean_absolute_percentage_error

from globals import global_sae_batch_size


def read_data_set_pair(features_file, labels_file):
    """

    Arguments:

        features_file -- csv file containing the features
        labels_file -- csv file containing the labels

    Returns:

        X - features as a numpy array
        Y - labels as a numpy array
    """

    data_frame_features = pa.read_csv(features_file, header=None)
    data_frame_labels = pa.read_csv(labels_file, header=None)


    return data_frame_features.values, data_frame_labels.values


def create_time_window_dataset(input_data, time_steps, batch_size):
    """
    Converts the pandas.DataFrame object in the right model input

    Arguments:
        input_data -- array containing the input data
        time_steps -- number of time steps to look back
        batch_size -- batch size

    Returns:
        data_X -- LSTM input with shape (samples, time_steps, features)
        data_Y -- labels for the data_X with shape (samples, features)
    """

    generator = TimeseriesGenerator(input_data, input_data, length=time_steps, batch_size=batch_size)

    data_X = []
    data_Y = []

    # convert to LSTM (samples, time_steps, features) format
    for i in range(len(generator)):

        batch_X, batch_Y = generator[i]
        if i == 0:
            data_X = batch_X
            data_Y = batch_Y
        else:
            data_X = np.concatenate([data_X, batch_X])
            data_Y = np.concatenate([data_Y, batch_Y])

    return data_X, data_Y


def load_and_transform_data(features_file, labels_file, time_steps, batch_size,
                            scale_type, scaler_X = None, scaler_Y = None,
                            wavelet_transform_iterations=0, encode_features=False,
                            adjust_to_multiple_of_batch_size=True):
    """
    Given a set of files for features and labels, it reads the content for each and returns a tuple X, Y to be fed to the model

    Arguments:
        features_file -- csv file containing the features
        labels_file -- csv file containing the labels
        time_steps  -- number of time steps to look back
        batch_size  -- batch size
        scale_type -- what scaler to use: none, normal or standard
        scaler_X -- if provided it uses it to scale the features, i.e. the X
        scaler_Y -- if provided it uses it to scale the labels, i.e. the Y
        wavelet_transform_iterations -- number of times to perform the wavelet single level discrete transformation
        encode_features -- whether to use the stacked auto encoders for the features
        adjust_to_multiple_of_batch_size -- reduces the samples number to divide to the batch_size without a remainder,
        which is required for a stateful LSTM

    Returns:
        data_X -- LSTM input with shape (samples, time_steps, features)
        data_Y -- labels for the data_X with shape (samples, features)
        scaler_X -- scaler used in transforming the features data
        scaler_Y -- scaler used in transforming the labels data

    """

    x, y = read_data_set_pair(features_file, labels_file)

    if scale_type == 'normal':
        normalize_input = True
    elif scale_type == 'standard':
        standardize_input = True

    if scaler_X == None and scaler_Y == None:
        if normalize_input == True:
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_X = scaler_X.fit(x)

            scaler_Y = MinMaxScaler(feature_range=(0, 1))
            scaler_Y = scaler_Y.fit(y)


        elif standardize_input == True:
            scaler_X = StandardScaler()
            scaler_X = scaler_X.fit(x)

            scaler_Y = StandardScaler()
            scaler_Y = scaler_Y.fit(y)

    if scaler_X != None and scaler_Y != None:
        x = scaler_X.transform(x)
        y = scaler_Y.transform(y)

    # Only the inputs are transformed. The labels, i.e. y, are kept untransformed
    for i in range(0, x.shape[1]):
        n = x[:, i].size
        for j in range(0, wavelet_transform_iterations):
            coefficients = pywt.wavedec(x[:, i], 'db2', mode='symmetric', level=None, axis=0)
            coefficients_transformed = []
            # get the aproximation coeffiecients which are not thresholded
            coefficients_transformed.append(coefficients[0])
            # transform the details coefficients by removing the ones more than a full standard deviation away
            for detail_coefficient in coefficients[1:]:
                coefficients_transformed.append(
                    pywt.threshold(detail_coefficient, np.std(detail_coefficient), mode='garrote'))

            temp_array = pywt.waverec(coefficients_transformed, 'db2', mode='symmetric', axis=0)

            x[:, i] = temp_array[:n]

    x, _ = create_time_window_dataset(x, time_steps=time_steps, batch_size=batch_size)
    _, y = create_time_window_dataset(y, time_steps=time_steps, batch_size=batch_size)

    if encode_features == True:

        x, _ = run_function_in_separate_process(generate_encoded_features_from_saved_model, x,
                                                'models', time_steps, scale_type,
                                                wavelet_transform_iterations)

    if adjust_to_multiple_of_batch_size == True:
        # adjust the number of samples to divide to the batch_size and have the model train faster
        adj_number_of_samples = x.shape[0] - x.shape[0] % batch_size
        x, y = x[:adj_number_of_samples, :, :], y[:adj_number_of_samples, :]

    return x, y, scaler_X, scaler_Y


def generate_encoded_features_from_saved_model(x, input_directory, time_steps, scale_type, wavelet_transform_iterations):
    """

    function that loads the right stacked autoencoder model based on the parameters given and generates an array of encoded features
    Arguments:

        x -- features to be encoded
        input_directory -- directory where the sae models are located
        time_steps -- time steps used to shape the data. Hyperparameter used when training the SAE model
        scale_type -- scale type used to transform the data. Hyperparameter used when training the SAE model
        wavelet_transform_iterations -- number of wavelet transformations performed on the features data. Hyperparameter used when training the SAE model

    Returns:

        encoded_X - encoded features using the SAE model loaded based on the parameters provided

    """

    model = load_stacked_autoencoder_model(input_directory=input_directory, time_steps=time_steps,
                                           scale_type=scale_type,
                                           wavelet_transformation_iterations=wavelet_transform_iterations)

    encoded_x = generate_encoded_features(model, x)

    return encoded_x

def save_model_with_additional_data(model, scaler_X, scaler_Y, output_directory):

    """
    
    Saves the model and additional data, like the scaler used
    
    Arguments:

        model -- model to be saved
        scaler_X -- scaler for the features to be dumped in a file
        scaler_Y -- scaler for the labels to be dumped in a file
    
    Returns:


    """

    # save the full model to h5 format
    model.save(output_directory + '/prediction_model.h5')
    print("Saved model to disk")

    # save the scalers in joblib format
    if scaler_X != None and scaler_Y != None:
        joblib.dump(scaler_X, output_directory + '/scaler_X.gz')
        joblib.dump(scaler_Y, output_directory + '/scaler_Y.gz')
        print("Saved scalers to disk")


def load_additional_data(output_directory):
    """

    Loads the model additional data, like the scalers used

    Arguments:

    Returns:

        scaler_X -- loaded scaler for features
        scaler_Y -- loaded scaler for labels

    """

    # load the scalers from joblib format
    scaler_file_name = output_directory + '/scaler_X.gz'
    if os.path.isfile(scaler_file_name) == True:
        scaler_X = joblib.load(scaler_file_name)

    scaler_file_name = output_directory + '/scaler_Y.gz'
    if os.path.isfile(scaler_file_name) == True:
        scaler_Y = joblib.load(scaler_file_name)

    print("Loaded scalers from disk")

    return scaler_X, scaler_Y


def load_prediction_model(output_directory):
    """

    Loads the model

    Arguments:

    Returns:

        model -- loaded model

    """

    # load the full model in h5 format
    model = load_model(output_directory + '/prediction_model.h5',
               custom_objects={'root_mean_square_error': root_mean_square_error,
                               'theil_u': theil_u,
                               'pearson_r': pearson_r,
                               'custom_epsilon_mean_absolute_percentage_error': custom_epsilon_mean_absolute_percentage_error})
    print("Loaded model from disk")

    return model


def load_stacked_autoencoder_model(input_directory, time_steps, scale_type, wavelet_transformation_iterations):
    """

    Loads the stacked autoencoder model based on the parameters given

    Arguments:

        input_directory -- directory where all trained SAE models are located
        time_steps -- time steps used to shape the data. Hyperparameter used when training the SAE model
        scale_type -- scale type used to transform the data. Hyperparameter used when training the SAE model
        wavelet_transform_iterations -- number of wavelet transformations performed on the features data. Hyperparameter used when training the SAE model

    Returns:

        model -- the desired SAE model

    """

    # load the full model in h5 format
    model = load_model(input_directory + '/sae_model_' + str(time_steps) + '_' + str(scale_type) + '_' + str(wavelet_transformation_iterations) + '.h5')

    print("Loaded stacked auto encoder model from disk")

    return model

def generate_encoded_features(model, x):
    """

    Generates the encoded features using the SAE model and original features

    Arguments:

        model -- the SAE model
        x -- features to be encoded

    Returns:

        predictions -- the encoded features

    """

    predictions = model.predict(x=x, batch_size=global_sae_batch_size)

    return predictions


def plot_real_vs_predicted_data(real_output, predicted_output):

    """
    Plots the real against predicted values with a bloomberg-like dark background

    Arguments:
        real_output  -- array with real data
        predicted_output  -- array with predicted data

    Returns:
    """

    # Plot real & predicted values
    params = plt.gcf()
    plSize = params.get_size_inches()
    params.set_size_inches((plSize[0] * 4, plSize[1] * 2))
    plt.plot(predicted_output, color='r')
    plt.plot(real_output, color='y')
    plt.title('Real vs Predicted')
    plt.ylabel('Relative performance')
    plt.xlabel('Days')
    plt.legend(['Predicted', 'Real'], loc='upper right')

    plt.show()

    plt.close()


def create_data_directory(output_directory, epochs=None, batch_size=None, stateful=None,
                          model_type=None, rnn_units=None, level1_units=None, level2_units=None, time_steps=None,
                          scale_type=None, wavelet_transform_iterations=None, encode_features=None):
    """
    Creates a timestamp and hyperparameters based directory if required

    Arguments:
        output_directory -- base directory where the folder should be created
        epochs -- value of the epochs hyperparameter used
        batch_size -- value of the batch_size hyperparameter used
        stateful -- whether the sequential model (GRU or LSTM) is stateful or not
        model_type -- the sequential model used: lstm or gru
        rnn_units -- value of the rnn_units hyperparameter used
        level1_units -- number of units in the 1st level used for the SAE model
        level2_units -- number of units in the 2nd level used for the SAE model
        time_steps -- value of the time_steps hyperparameter used
        scale_type -- value of the scale_type hyperparameter used
        wavelet_transform_iterations -- value of the wavelet_transform_iterations hyperparameter used
        encode_features -- value of the encode_features hyperparameter used


    Returns:

        data_folder_name    -- directory in the following format: YYYYMMDDHHMMSS_timesteps_batchsize_stateful_epochs_modeltype_units_level1units_level2units_scaletype_wavelettransformations_encodefeatures
    """

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

    data_folder_name = output_directory + "/" + timestamp

    if time_steps != None:
        data_folder_name = data_folder_name + "_" + str(time_steps)

    if batch_size != None:
        data_folder_name = data_folder_name + "_" + str(batch_size)

    if stateful != None:
        data_folder_name = data_folder_name + "_" + str(stateful)

    if epochs != None:
        data_folder_name = data_folder_name + "_" + str(epochs)

    if model_type != None:
        data_folder_name = data_folder_name + "_" + str(model_type)

    if rnn_units != None:
        data_folder_name = data_folder_name + "_" + str(rnn_units)

    if level1_units != None:
        data_folder_name = data_folder_name + "_" + str(level1_units)

    if level2_units != None:
        data_folder_name = data_folder_name + "_" + str(level2_units)

    if scale_type != None:
        data_folder_name = data_folder_name + "_" + str(scale_type)

    if wavelet_transform_iterations != None:
        data_folder_name = data_folder_name + "_" + str(wavelet_transform_iterations)

    if encode_features != None:
        data_folder_name = data_folder_name + "_" + str(encode_features)

    os.makedirs(data_folder_name)

    return data_folder_name

def wrapper_func(func, queue, *args):

    """

    Wrapper over a function that will be called in a different process

    Arguments:

         func -- function name to be called
         queue -- queue to store the results and errors
         *args -- arguments to be passed to the function func

    Returns:


    """

    try:
        result = func(*args)
        error = None
    except Exception:
        result = None
        ex_type, ex_value, tb = sys.exc_info()
        error = ex_type, ex_value,''.join(traceback.format_tb(tb))
    queue.put((result, error))


def process(func, *args):
    """

    Creates a separate process which will run a specific function and waits until the process finishes

    Arguments:

        func -- function name to be called
        *args -- arguments to be passed to the function func

    Returns:

        result -- the return variables of the target function
        error -- errors that happened during the execution


    """

    queue = Queue()
    p = Process(target = wrapper_func, args = [func] + [queue] + list(args))
    p.start()
    result, error = queue.get()
    p.join()
    return result, error


def run_function_in_separate_process(func, *args):

    """

    Runs a specific function with the given parameters in a different process

    Arguments:

        func -- function name to be called
        *args -- arguments to be passed to the function func

    Returns:

        result -- the return variables of the target function
        error -- errors that happened during the execution


    """

    result, error = process(func, *args)
    return result, error

def compute_score(y_pred, r):

    """

    Computes the score according to the Kaggle competition: https://www.kaggle.com/c/two-sigma-financial-news#evaluation

    Arguments:

        y_pred -- predictions. Values scaled to be between [-1, 1]
        r -- relative market return

    Returns:

        score - the computed score


    """

    x_t = y_pred * r

    score = np.mean(x_t) / np.std(x_t)

    return score

def compute_profitability(y_true, y_pred):
    """

    Computes a buy_or_sell trading stategy based on the predictions.

    Arguments:

        y_pred -- predictions.
        y_true -- real values

    Returns:

        score - the computed score

    """

    length_y = len(y_true)
    strategy_earnings = 0
    tran_cost_rate = 0.0001

    for i in range(length_y - 1):
        if y_pred[i + 1] < y_true[i]:
            difference = y_true[i] - y_true[i + 1]
        elif y_pred[i + 1] >= y_true[i]:
            difference = y_true[i + 1] - y_true[i]

        strategy_earnings += (difference + tran_cost_rate * (y_true[i + 1] + y_true[i])) / y_true[i]

    strategy_earnings = 100 * strategy_earnings

    return strategy_earnings

