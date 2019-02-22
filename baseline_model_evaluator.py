import numpy as np

from keras.models import load_model
from keras_extensions import root_mean_square_error

from globals import global_time_steps, global_batch_size, global_epochs, global_lstm_units, base_dir
from utils import plot_real_predicted_data, loadLSTMLabeledData
from baseline_model import SingleTraining


def TrainAndGenerateDataWithPredefinedGlobalHyperparameters():
    """
    Once we have a set of hyperparameters that produced the best results
    we have to check how they perform on the test data. So we set them in the globals.py
    and this function trains again the model with the same parameters, saves it and runs it on the test set

    Arguments:

    Returns:

    """
    model, input_directory = SingleTraining(time_steps=global_time_steps, batch_size=global_batch_size, epochs=global_epochs,
                                            lstm_units=global_lstm_units, just_training=False, output_directory=None)

    LoadAndEvaluateModelOnTestData(model, input_directory)

def LoadAndEvaluateModelOnTestData(model, train_data_directory, load_model_from_file=False):

    """
    Once we have a set of hyperparameters that produced the best results
    we have to check how they perform on the test data. So we set them in the globals.py
    and this function trains again the model with the same parameters, saves it and runs it on the test set

    Arguments:

        model -- the trained model that should be evaluated on the test data
        train_data_directory -- where the model and plots were saved
        load_model_from_file -- if set to True it loads the model from the file

    Returns:

    """

    # global_time_steps and global_batch_size must be set MANUALLY
    # to the values that the model was trained on before running this python script
    test_x, test_y = loadLSTMLabeledData('data/test.csv', time_steps=global_time_steps, batch_size=global_batch_size,
                                         adjust_for_batches=True, plot_data=True,
                                         output_directory=train_data_directory, file_name='TestDataPlot')

    if load_model_from_file == True:
        model = load_model(train_data_directory + '/model_baseline.h5',
                           custom_objects={'root_mean_square_error': root_mean_square_error})

    scores = model.evaluate(x=test_x, y=test_y, batch_size=global_batch_size)

    rmse_test = scores[1]

    with open(train_data_directory + '/stats.txt', 'a') as f:
        f.write("%s: Test - %.6f  Time_steps: %d Batch_size=%d Epochs: %d LSTM units: %d \n" %
                (model.metrics_names[1], rmse_test, global_time_steps, global_batch_size, global_epochs, global_lstm_units))

    predictions = model.predict(x=test_x, batch_size=global_batch_size)

    plot_real_predicted_data(test_y, predictions, train_data_directory, "RealVsPredicted", show_plot=False,
                             save_plot=True)

if __name__ == '__main__':

    TrainAndGenerateDataWithPredefinedGlobalHyperparameters()


