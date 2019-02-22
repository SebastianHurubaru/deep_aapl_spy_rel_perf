# define the hyperparameters

# this group cannot be changed when applying GridSearch
features = 1                      # we have only one return


global_time_steps = 10                   # the number of lag returns that should predict the next one
global_batch_size = 128
global_epochs = 500
global_lstm_units = 200

base_dir = 'output'

# define other not training related global parameters
number_of_physical_cores = 16
number_of_gpus = 2
