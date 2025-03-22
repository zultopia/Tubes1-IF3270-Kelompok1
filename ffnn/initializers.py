import numpy as np

def zero_init(shape):
    return np.zeros(shape)

def uniform_init(shape, lower=-0.1, upper=0.1, seed=None):
    np.random.seed(seed)
    return np.random.uniform(lower, upper, shape)

def normal_init(shape, mean=0.0, variance=0.1, seed=None):
    np.random.seed(seed)
    return np.random.normal(mean, variance, shape)
