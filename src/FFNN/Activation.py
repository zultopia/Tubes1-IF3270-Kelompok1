import numpy as np

class Activation:
    @staticmethod
    def linear(x): 
        return x

    @staticmethod
    def d_linear(x): 
        return np.ones_like(x)

    @staticmethod
    def relu(x): 
        return np.maximum(0, x)
    
    @staticmethod
    def d_relu(x): 
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x): 
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def d_sigmoid(x):
        sig = Activation.sigmoid(x)
        return sig * (1 - sig)
    
    @staticmethod
    def tanh(x): 
        return np.tanh(x)
    
    @staticmethod
    def d_tanh(x): 
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def d_softmax(x):
        s = Activation.softmax(x) 
        batch_size, n = s.shape
        jacobian = np.zeros((batch_size, n, n))  

        for b in range(batch_size):
            s_b = s[b, :].reshape(-1, 1) 
            jacobian[b] = np.diagflat(s_b) - np.dot(s_b, s_b.T)  

        return jacobian

