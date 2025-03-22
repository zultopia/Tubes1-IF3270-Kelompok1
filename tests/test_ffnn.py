import numpy as np
from ffnn.model import FFNN
from ffnn.layers import Layer
from ffnn.activations import relu
from ffnn.initializers import zero_init

def test_forward_pass():
    layer1 = Layer(3, 2, relu, zero_init)
    model = FFNN([layer1])
    X = np.array([[1, 2, 3]])
    assert model.forward(X).shape == (1, 2)

def test_loss():
    from ffnn.losses import mse
    y_true = np.array([[1, 0]])
    y_pred = np.array([[0.5, 0.5]])
    assert mse(y_true, y_pred) > 0

if __name__ == "__main__":
    test_forward_pass()
    test_loss()
    print("All tests passed!")