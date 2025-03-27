import numpy as np

class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    @staticmethod
    def d_mse(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-8):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def d_binary_cross_entropy(y_true, y_pred, epsilon=1e-8):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-8):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def d_categorical_cross_entropy(y_true, y_pred, epsilon=1e-8):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred
