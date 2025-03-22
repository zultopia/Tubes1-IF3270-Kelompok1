from ffnn import FFNN, Layer, relu, uniform_init, Trainer, mse, d_mse
import numpy as np

# Example
X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
y = np.array([[1, 0], [0, 1]])

layer1 = Layer(3, 5, relu, lambda shape: uniform_init(shape, -0.5, 0.5))
layer2 = Layer(5, 2, relu, lambda shape: uniform_init(shape, -0.5, 0.5))

model = FFNN([layer1, layer2])
trainer = Trainer(model, mse, d_mse, lr=0.01)
trainer.train(X, y, epochs=10)