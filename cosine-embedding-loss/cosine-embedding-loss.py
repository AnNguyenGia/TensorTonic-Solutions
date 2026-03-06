import numpy as np
from numpy import linalg as LA
def cosine(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.dot(x1, x2.T)/(LA.norm(x1)*LA.norm(x2))

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    
    if label == 1:
        L = 1-cosine(x1, x2)
    else:
        L = max(0, cosine(x1, x2)-margin)
    return L
  
    