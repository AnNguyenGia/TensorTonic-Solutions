import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if len(y_pred) != len(y_true):
        return None
    sum = 0
    for x, y in zip(y_pred, y_true):
        sum += (x-y)**2
    return sum/len(y_true)
