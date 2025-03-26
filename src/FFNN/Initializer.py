import numpy as np

class Initializer:
    @staticmethod
    def zero(shape): 
        return np.zeros(shape)
    
    @staticmethod
    def uniform(shape, lower=-0.5, upper=0.5, seed=None):
        if seed: 
            np.random.seed(seed)
        return np.random.uniform(lower, upper, shape)
    
    @staticmethod
    def normal(shape, mean=0.0, variance=0.1, seed=None):
        if seed:
            np.random.seed(seed)
        return np.random.normal(mean, np.sqrt(variance), shape)

    @staticmethod
    def xavier(shape, seed=None):
        if seed:
            np.random.seed(seed)
        d = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-d, d, shape)
    
    @staticmethod
    def he(shape, seed=None):
        if seed:
            np.random.seed(seed)
        d = np.sqrt(2 / shape[0])
        return np.random.normal(0, d, shape)
