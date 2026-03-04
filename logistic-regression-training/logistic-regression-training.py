import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w = np.zeros(X.shape[1])
    n = X.shape[0]
    h = 0

    res_w = None
    res_h = None
    minLoss = 100
    for _ in range(steps):
        z = np.dot(X, w.T) + h
        p = _sigmoid(z)
        loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        if loss < minLoss:
            minLoss = loss
            res_w = w
            res_h = h
        update_w = np.dot(X.T, p-y)/n
        update_h = np.mean(p-y)
    #     # print(z)
        w -= lr * update_w
        h -= lr * update_h

    return (res_w, res_h)
    return w, h
    