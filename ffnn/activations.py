import numpy as np

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Derivatives
def d_linear(x):
    return np.ones_like(x)

def d_relu(x):
    return (x > 0).astype(float)

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def d_softmax(x):
    s = softmax(x)
    return s * (1 - s)