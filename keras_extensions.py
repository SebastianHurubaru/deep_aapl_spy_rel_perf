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

def custom_epsilon_mean_absolute_percentage_error(y_true, y_pred):
    """
        Creates the Root Mean Square Error implemented using Mean Square Error

        Arguments:
            y_true -- real values
            y_pred  -- predicted values

        Returns:

            root mean square error of y_true and y_pred
        """

    diff = K.backend.abs((y_true - y_pred) / K.backend.clip(K.backend.abs(y_true), 1, None))
    return 100. * K.backend.mean(diff, axis=-1)



# Root Mean Square Error (RMSE)
def theil_u(y_true, y_pred):

    """
    Creates the Root Mean Square Error implemented using Mean Square Error

    Arguments:
        y_true -- real values
        y_pred  -- predicted values

    Returns:

        root mean square error of y_true and y_pred
    """

    theil_u = root_mean_square_error(y_true, y_pred)/\
           (K.backend.sqrt(K.backend.mean(K.backend.square(y_true), axis=-1)) + K.backend.sqrt(K.backend.mean(K.backend.square(y_pred), axis=-1)))

    return theil_u

# Root Mean Square Error (RMSE)
def R(y_true, y_pred):

    """
    Creates the Root Mean Square Error implemented using Mean Square Error

    Arguments:
        y_true -- real values
        y_pred  -- predicted values

    Returns:

        root mean square error of y_true and y_pred
    """

    y_true_mean = K.backend.mean(y_true, axis=-1)
    y_pred_mean = K.backend.mean(y_pred, axis=-1)

    r = K.backend.sum((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=-1)/\
        K.backend.sqrt(K.backend.sum(K.backend.square(y_true - y_true_mean) * K.backend.square(y_pred - y_pred_mean), axis=-1))

    return r
