# Apple relative performance to S&P 500 model prediction with Keras

CS230 Winter 2019: Prediction of Apple shares relative relative performance to SPY

We have generated in total a number of eight models:
  -  GRU/LSTM: a single layer of GRU/LSTM with 128 units
  -  Wavelet GRU/Wavelet LSTM: the previous model with performing two wavelet transformation on the features
  -  SAE Wavelet GRU/SAE Wavelet LSTM: the model above with encoding the twenty features in only ten deep features 
  -  Tweets SAE Wavelet GRU/Tweets SAE Wavelet LSTM: the model above with adding extracted sentiment from the daily tweets as feature
    

The project is structured in the followings:

  - /data: folder containing the datasets in original and then pre-processed to fit the models
  - /output: contains automatically created folders with the some models and their associated scalers
  - prediction_model.py: file containing the implementation of the models enumerated above
  - prediction_model_generator.py: generates the model file for the hyperparameters specified in the globals.py file
  - globals.py: file containing the hyperparameters and the rest of the global values
  - keras_extensions.py: contains variouse custom metric functions: RMSE, Theil U, MAPE(customized with an epsilon value of 1)
  - utils.py: file containing utility functions used throughout the project
  - sae_model.py: file containing the implementation of the Stacked Auto Encoders models
  - Predicting_Movie_Reviews_with_BERT_on_TF_Hub-Modified.py: sample file from Google that trains BERT on a movie review dataset
  - bert_sentiment_predictor.py: file containing the sentiment extraction from a daily list of tweets using BERT
  
  
  
  
