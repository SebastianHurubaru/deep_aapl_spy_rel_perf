import numpy as np
import time
import datetime
import os
import pandas as pa
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator


def read_csv_file(file_name):
    """
    Reads a csv file using pandas module

    Arguments:
    file_name -- path to the file

    Returns:
    data_frame -- pandas.DataFrame object
    """

    data_frame = pa.read_csv(file_name)

    return data_frame

def create_time_window_dataset(data_frame, time_steps, batch_size):
    """
    Converts the pandas.DataFrame object to a proper LSTM layer input

    Arguments:
        data_frame -- pandas.DataFrame object containing the data
        time_steps -- number of time steps to look back
        batch_size -- batch size

    Returns:
        data_X -- LSTM input with shape (samples, time_steps, features)
        data_Y -- labels for the data_X with shape (samples, features)
    """

    generator = TimeseriesGenerator(data_frame.values, data_frame.values, length=time_steps, batch_size=batch_size)

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


def loadLSTMLabeledData(input_file, time_steps, batch_size, adjust_for_batches=True, plot_data=False,
                        output_directory=None, file_name=None):
    """
    Given a file, it reads the content and returns a tuple X, Y to be given to the model

    Arguments:
        input_file  -- path to the input file
        time_steps  -- number of time steps to look back
        batch_size  -- batch size
        adjust_for_batches -- reduces the samples number to divide to the batch_size without a remainder,
        which is required for a stateful LSTM
        plot_data   -- decide whether to plot the data and save it as a jpeg image
        output_directory    -- works only when plot_data is True.
        It provides the location where the plot should be saved
        file_name   -- works only when plot_data is True.
        It specifies the jpeg file name

    Returns:
        data_X -- LSTM input with shape (samples, time_steps, features)
        data_Y -- labels for the data_X with shape (samples, features)
    """

    training_dataframe = read_csv_file(input_file)
    x, y = create_time_window_dataset(training_dataframe, time_steps=time_steps, batch_size=batch_size)

    if adjust_for_batches == True:
        # adjust the number of samples to divide to the batch_size and have the model train faster
        adj_number_of_samples = x.shape[0] - x.shape[0] % batch_size
        x, y = x[:adj_number_of_samples, :, :], y[:adj_number_of_samples, :]

    if plot_data == True:
        plot_dataset(training_dataframe.values, output_directory, file_name, show_plot=False, save_plot=True)

    return x, y

def plot_dataset(data_set, output_directory, file_name, show_plot, save_plot):
    """
    Plots a data set with a bloomberg-like dark background

    Arguments:
        data_set  -- data set to be plotted
        output_directory -- works only when save_plot is True.
        It specifies the location where the plot is going to be saved
        file_name   -- Works only when save_plot is set to True
        It specifies the jpeg file name containing the plot
        show_plot   -- whether the plot should be displayed or not
        save_plot   -- whether the plot should be saved to a file or not


    Returns:
    """

    # Plot training & validation loss values
    with plt.style.context(('dark_background')):
        params = plt.gcf()
        plSize = params.get_size_inches()
        params.set_size_inches((plSize[0] * 4, plSize[1]))
        plt.plot(data_set)
        plt.title('AAPL to SPY relative performance')
        plt.ylabel('Price')
        plt.xlabel('Time')

        if save_plot == True:
            plt.savefig(output_directory + '/' + file_name + '.jpg', format='jpg', quality=95, dpi=200)

        if show_plot == True:
            plt.show()

    plt.close()


def generate_train_dev_plots(history, show_plot, save_plot, output_directory):
    """
    Plots the train and dev losses data with a bloomberg-like dark background

    Arguments:
        history  -- model.fit output history data
        show_plot   -- whether the plot should be displayed or not
        save_plot   -- whether the plot should be saved to a file or not
        output_directory -- works only when save_plot is True.
        It specifies the location where the plot is going to be saved
        file_name   -- Works only when save_plot is set to True
        It specifies the jpeg file name containing the plot

    Returns:
    """

    # Plot training & validation loss values
    with plt.style.context(('dark_background')):
        params = plt.gcf()
        plSize = params.get_size_inches()
        params.set_size_inches((plSize[0] * 4, plSize[1]))
        plt.plot(history.history['root_mean_square_error'])
        plt.plot(history.history['val_root_mean_square_error'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Dev'], loc='upper right')

        if save_plot == True:
            plt.savefig(output_directory + '/TrainDevLossesPlot.jpg', format='jpg', quality=95, dpi=200)

        if show_plot == True:
            plt.show()

    plt.close()

def plot_real_predicted_data(real_output, predicted_output, output_directory, file_name, show_plot, save_plot):

    """
    Plots the real against predicted values with a bloomberg-like dark background

    Arguments:
        real_output  -- array with real data
        predicted_output  -- array with predicted data
        output_directory -- works only when save_plot is True.
        file_name   -- Works only when save_plot is set to True
        It specifies the jpeg file name containing the plot
        It specifies the location where the plot is going to be saved
        show_plot   -- whether the plot should be displayed or not
        save_plot   -- whether the plot should be saved to a file or not

    Returns:
    """

    # Plot real & predicted values
    with plt.style.context(('dark_background')):
        params = plt.gcf()
        plSize = params.get_size_inches()
        params.set_size_inches((plSize[0] * 4, plSize[1] * 2))
        plt.plot(predicted_output, color='y')
        plt.plot(real_output, color='r')
        plt.title('Real vs Predicted')
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.legend(['Predicted', 'Real'], loc='upper right')

        if save_plot == True:
            plt.savefig(output_directory + '/' + file_name + '.jpg', format='jpg', quality=95, dpi=200)

        if show_plot == True:
            plt.show()

    plt.close()


def create_data_directory(output_directory, epochs=None, batch_size=None, lstm_units=None, time_steps=None):
    """
    Creates a timestamp and hyperparameters based directory if required

    Arguments:
        output_directory -- base directory where the folder should be created
        epochs -- value of the epochs hyperparameter used
        batch_size -- value of the batch_size hyperparameter used
        lstm_units -- value of the lstm_units hyperparameter used
        time_steps -- value of the time_steps hyperparameter used


    Returns:

        data_folder_name    -- directory in the following format: YYYYMMDDHHMMSS_timesteps_batchsize_epochs_lstmunits
    """

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

    data_folder_name = output_directory + "/" + timestamp

    if time_steps != None:
        data_folder_name = data_folder_name + "_" + str(time_steps)

    if batch_size != None:
        data_folder_name = data_folder_name + "_" + str(batch_size)

    if epochs != None:
        data_folder_name = data_folder_name + "_" + str(epochs)

    if lstm_units != None:
        data_folder_name = data_folder_name + "_" + str(lstm_units)

    os.makedirs(data_folder_name)

    return data_folder_name
