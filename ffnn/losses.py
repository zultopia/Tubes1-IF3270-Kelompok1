import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def d_mse(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def d_binary_cross_entropy(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def d_categorical_cross_entropy(y_true, y_pred):
    return y_pred - y_true