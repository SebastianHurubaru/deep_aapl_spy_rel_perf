import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from prediction_model import SingleEvaluation
from utils import plot_real_vs_predicted_data, compute_score, compute_profitability


if __name__ == '__main__':

    with tf.device('/cpu:0'):

        # set the global variables in the globals.py file to the ones that generated the best results
        # copy the model and scaler files in the models directory

        # run the model to predict the data and generate some plots.
        train_Y, train_Y_hat = SingleEvaluation('data/train_X.csv', 'data/train_Y.csv')
        #plot_real_vs_predicted_data(train_Y, train_Y_hat)

        dev_test_Y, dev_test_Y_hat = SingleEvaluation('data/dev_test_X.csv', 'data/dev_test_Y.csv')
        #plot_real_vs_predicted_data(dev_test_Y, dev_test_Y_hat)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train_Y)

        scaled_dev_test_Y_hat = scaler.transform(dev_test_Y_hat)

        print("Kaggle score: " + str(compute_score(scaled_dev_test_Y_hat, dev_test_Y)))
        print("Profitability: " + str(compute_profitability(dev_test_Y, dev_test_Y_hat)))




