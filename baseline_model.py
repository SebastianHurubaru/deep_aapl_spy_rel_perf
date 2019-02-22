import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.utils import multi_gpu_model
from keras_extensions import root_mean_square_error
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from globals import features, global_time_steps, global_batch_size, global_epochs, global_lstm_units, base_dir, number_of_gpus
from utils import generate_train_dev_plots, create_data_directory, loadLSTMLabeledData


def create_baseline_model(time_steps, batch_size, lstm_units):
    """
    Creates the baseline model consisting of a LSTM layer
    followed by a BatchNormalization one yielding a better performance

    Arguments:
        time_steps -- value of the time_steps hyperparameter used, i.e. the number of time steps to look back
        batch_size -- value of the batch_size hyperparameter used
        lstm_units -- value of the lstm_units hyperparameter used, i.e. the number of LSTM units in the LSTM layer

    Returns:

        model    -- Keras model
    """
    with tf.device('/cpu:0'):

        # define model
        model = Sequential()
        model.add(LSTM(lstm_units, activation='tanh', kernel_initializer='normal',
                       input_shape=(time_steps, features),
                       batch_input_shape=(batch_size, time_steps, features),
                       stateful=True,
                       return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dense(features, kernel_initializer='normal', activation='linear'))


    # Replicates the model on the number of given GPUs.
    # This assumes that the machine has that specified number of available GPUs.
    # Keras bug: state does not get properly replicated on each GPU, leading to shape mismatch
    #parallel_model = multi_gpu_model(model, gpus=number_of_gpus)

    model.compile(optimizer='adam', loss='mse', metrics=[root_mean_square_error])
    model.summary()

    return model

def GridSearchTraining(model, X, Y):
    """
    Grid Search Hyperparameters tunning

    Arguments:
        model -- the Keras model
        X -- model input
        Y -- input labels

    Returns:

        prints statistics in the output console
    """

    epochs = [10, 20, 40, 60, 80, 100]
    lstm_units = [100, 200, 400, 600, 800, 1000]


    model = KerasClassifier(build_fn=create_baseline_model)

    param_grid = dict(epochs=epochs, lstm_units=lstm_units)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)

    grid_result = grid.fit(X, Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def Training(model, train_X, train_Y, dev_X, dev_Y,
             time_steps, batch_size, epochs, lstm_units,
             output_directory, just_training=False):
    """
    Trains the model, evaluates it separately on the train and dev set to get
    the rmse statistic for both sets which are printed in an output file.
    If called from a standalone context it saves all the plots and the model with the weights

    Arguments:
        model       -- Keras model
        train_X     -- training input data
        train_Y     -- labels for the training input data
        dev_X       -- dev input data
        dev_Y       -- labels for the dev input data

        time_steps -- value of the time_steps hyperparameter used, i.e. the number of time steps to look back
        batch_size -- value of the batch_size hyperparameter used
        epochs     -- value of the epcchs hyperparameter used, i.e. how many times should we go through our whole training set
        lstm_units -- value of the lstm_units hyperparameter used, i.e. the number of LSTM units in the LSTM layer

        output_directory -- directory where to save plot and statistics data
        just_training    -- flag to indicate that we are just training and we do not want to save the plots and the model.
        Used basically when searching for the hyperparameters outputing the best results

    Returns:

    """

    # fit model
    history = model.fit(x=train_X,
                        y=train_Y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(dev_X, dev_Y))

    scores = model.evaluate(x=train_X, y=train_Y, batch_size=batch_size)
    rmse_train = scores[1]

    scores = model.evaluate(x=dev_X, y=dev_Y, batch_size=batch_size)
    rmse_dev = scores[1]

    with open(output_directory + '/stats.txt', 'a') as f:
        f.write("%s: Train - %.6f  Dev - %.6f Time_steps: %d Batch_size=%d Epochs: %d LSTM units: %d \n" %
                (model.metrics_names[1], rmse_train, rmse_dev, time_steps, batch_size, epochs, lstm_units))

    if just_training == False:
        generate_train_dev_plots(history=history, show_plot=True, save_plot=True, output_directory=output_directory)

        # save the full model to h5 format
        model.save(output_directory + '/model_baseline.h5')
        print("Saved model to disk")

def SingleTraining(time_steps, batch_size, epochs, lstm_units, output_directory, just_training=True):
    """
    Wrapper function over Training that previously loads the training and dev sets
    creates the output directory based on hyperparameters value if requested and
    calls the Training function

    Arguments:

        time_steps -- value of the time_steps hyperparameter used, i.e. the number of time steps to look back
        batch_size -- value of the batch_size hyperparameter used
        epochs     -- value of the epcchs hyperparameter used, i.e. how many times should we go through our whole training set
        lstm_units -- value of the lstm_units hyperparameter used, i.e. the number of LSTM units in the LSTM layer

        output_directory -- directory where to save plot and statistics data.
        If just_training parameter is False the directory is created here using the local hyperparameter values
        just_training    -- flag to indicate that we are just training and we do not want to save the plots and the model.
        Used basically when searching for the hyperparameters outputing the best results

    Returns:

        model       -- Keras model
        output_directory -- where the model and data was written

    """

    if just_training == False:
        output_directory = create_data_directory(base_dir, time_steps=time_steps, batch_size=batch_size,
                                                 epochs=epochs, lstm_units=lstm_units)

    train_x, train_y = loadLSTMLabeledData('data/train.csv', time_steps=time_steps, batch_size=batch_size,
                                           adjust_for_batches=True, plot_data=True,
                                           output_directory=output_directory, file_name='TrainDataPlot')

    dev_x, dev_y = loadLSTMLabeledData('data/dev.csv', time_steps=time_steps, batch_size=batch_size,
                                       adjust_for_batches=True, plot_data=True,
                                       output_directory=output_directory, file_name='DevDataPlot')

    model = create_baseline_model(time_steps=time_steps, batch_size=batch_size, lstm_units=lstm_units)

    Training(model, train_x, train_y, dev_x, dev_y,
             time_steps=time_steps, batch_size=batch_size, epochs=epochs, lstm_units=lstm_units,
             just_training=just_training, output_directory=output_directory)

    return model, output_directory

def MultiSequentialTraining():
    """
    Wrapper function over SingleTraining that is a manual implementation of Grid Search
    Each Hyperparameter is actually an array and the model is trained on each combination.
    The statistic rmse is saved in a file for each run so that we can see the best results

    Arguments:

    Returns:

    """
    data_directory = create_data_directory(base_dir)

    time_steps_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    batch_size_list = [32, 64, 128]
    epochs_list = [10, 20, 40, 60, 80, 100]
    lstm_units_list = [25, 50]

    for time_steps in time_steps_list:
        for batch_size in batch_size_list:
            for epochs in epochs_list:
                for lstm_units in lstm_units_list:

                    SingleTraining(time_steps=time_steps, batch_size=batch_size,
                                         epochs=epochs, lstm_units=lstm_units, output_directory=data_directory)

if __name__ == '__main__':

    #Not working properly when LSTM is stateful, due to the batch_size
    #GridSearchTraining(model, train_dev_x, train_dev_y)

    MultiSequentialTraining()


