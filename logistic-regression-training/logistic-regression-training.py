import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code where
    w = np.zeros(X.shape[1])
    h = 0
    n = X.shape[0]
    minLoss = 100
    W = H = None
    for _ in range(steps):
        z = np.dot(X, w.T) + h
        p = _sigmoid(z)
        update_w = np.dot(p-y, X)/n
        update_h = np.mean(p-y)
        loss = -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
        w -= update_w*lr
        h -= update_h*lr
        if loss < minLoss:
            W = w
            H = h
        

    return W, H