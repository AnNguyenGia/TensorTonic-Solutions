import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    # temp = np.asarray(x, dtype=float)
    # res = [1/(1+np.exp(-x)) for x in temp]
   
    # if len(res) == 1:
    #     return res[0]
    # return np.array(res)
    if isinstance(x, int) or isinstance(x, float):
        return 1/(1+np.exp(-x))
    temp = np.asarray(x, dtype=float)
    res = [1/(1+np.exp(-x)) for x in temp]
    return np.array(res)
    