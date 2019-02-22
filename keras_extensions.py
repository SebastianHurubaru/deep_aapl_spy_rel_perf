import keras as K

# Root Mean Square Error (RMSE)
def root_mean_square_error(y_true, y_pred):

    """
    Creates the Root Mean Square Error implemented using Mean Square Error

    Arguments:
        y_true -- real values
        y_pred  -- predicted values

    Returns:

        root mean square error of y_true and y_pred
    """
    return K.backend.sqrt(K.losses.mean_squared_error(y_true, y_pred))

