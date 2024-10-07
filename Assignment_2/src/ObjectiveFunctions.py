import numpy as np

def absolute(x):
    return np.sum(np.abs(x))

def alpine(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1* x))