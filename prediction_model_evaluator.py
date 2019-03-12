from prediction_model import SingleTrainingWithGlobalVariables, SingleEvaluation

if __name__ == '__main__':

    # set the global variables in the globals.py file to the ones that generated the best results

    # run the training again on the best configuration to generate the model file and associated data
    # can be commented out once the files are generated
    #SingleTrainingWithGlobalVariables()

    # copy the model and scaler files in the models directory

    # run the model to predict the data and generate some plots.
    #SingleEvaluation('data/train_X.csv', 'data/train_Y.csv')
    SingleEvaluation('data/dev_test_X.csv', 'data/dev_test_Y.csv')