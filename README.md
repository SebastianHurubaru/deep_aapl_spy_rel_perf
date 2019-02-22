# Apple relative performance to S&P 500 model prediction with Keras

CS230 Winter 2019: Prediction of Apple shares relative relative performance to SPY

The baseline model contains a LSTM model followed by a BatchNormalization one for performance.

The project is structured in the followings

  - /data: folder containing the datasets. Original file plus converted files for train, dev and test were created
  - /output: contains automatically created folders with all required metrics and plots
  - baseline_model.py: file containing the implementation of the baseline model
  - baseline_model_evaluator.py: file containing the evaluation of the model configured with a certain set of hyperparameters on the test set
  - globals.py: file containing the hyperparameters and the rest of the global values
  - keras_extensions.py: contains a custom loss function, rmse, used as a metric
  - utils.py: file containing utility functions
  
  
  
