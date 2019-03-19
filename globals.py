# define the hyperparameters

# this group cannot be changed when applying GridSearch
features = 20                      # we have 20 features. See the original data for more details
encoded_features = 10              # generate a number of 10 encoded features
labels = 1                         # we have only one label, i.e. we are predicting the relative performance
tweets_sentiment_feature_index = 19 # the index of the sentiment feature

global_sae_batch_size = 64


global_time_steps = 1                   # the number of lag returns that should predict the next one
global_batch_size = 64
global_stateful = False
global_epochs = 5000
global_model_type = 'lstm'
global_units = 128
global_dropout_rate = 0.15
global_scale_type = 'normal'
global_wavelet_transform_iterations = 1
global_encode_features = True
global_include_tweets_sentiment = True

base_dir = 'output'

# define other not training related global parameters
number_of_physical_cores = 16
number_of_gpus = 2
