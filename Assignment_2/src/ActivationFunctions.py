import numpy as np

def linear(x, m = 1):
    return x * m

def step(x, g1 = 1, g2 = -1):
    if (x >= 0):
        return g1
    return g2

def ramp(x, g = 1, e = 1):
    if (x >= e):
        return g
    elif (x <= -e):
        return -g
    return x

def sigmoid(x, m = 1):
    return (1 / (1 + np.exp(-m * x)))

def hyperbolicTangent(x ,m = 1):
    return (2 / (1 + np.exp(-m * x))) - 1

def gaussian(x, std = 1):
    return np.exp((np.power(-x, 2)) / (np.power(std, 2)))

def relu(x):
    return np.maximum(0, x)