import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
        return s * (1 - s)

class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    @staticmethod
    def d_mse(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def d_binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def d_categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred

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

class FFNN:
    def __init__(self, layers, activations, loss, init_method='uniform', **init_params):
        self.layers = layers
        self.activations = [getattr(Activation, act) for act in activations]
        self.d_activations = [getattr(Activation, 'd_' + act) for act in activations]
        self.loss = getattr(Loss, loss)
        self.d_loss = getattr(Loss, 'd_' + loss)
        
        self.weights = []
        self.biases = []
        self.d_weights = []
        self.d_biases = []
        
        init_func = getattr(Initializer, init_method)
        for i in range(len(layers) - 1):
            self.weights.append(init_func((layers[i], layers[i+1]), **init_params))
            self.biases.append(init_func((1, layers[i+1]), **init_params))
            self.d_weights.append(np.zeros_like(self.weights[-1]))
            self.d_biases.append(np.zeros_like(self.biases[-1]))
    
    def forward(self, X):
        self.a = [X]
        self.z = []
        
        for i in range(len(self.weights)):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            a = self.activations[i](z)
            self.z.append(z)
            self.a.append(a)
        
        return self.a[-1]
    
    def backward(self, X, y):
        m = y.shape[0]
        if self.activations[-1] == Activation.softmax:
            dz = self.a[-1] - y
        else:
            dz = self.d_loss(y, self.a[-1]) * self.d_activations[-1](self.z[-1])

        for i in reversed(range(len(self.weights))):
            self.d_weights[i] = self.a[i].T @ dz / m
            self.d_biases[i] = np.sum(dz, axis=0, keepdims=True) / m
            
            # Clipping gradien
            self.d_weights[i] = np.clip(self.d_weights[i], -1, 1)
            self.d_biases[i] = np.clip(self.d_biases[i], -1, 1)
            
            if i > 0:
                dz = dz @ self.weights[i].T * self.d_activations[i-1](self.z[i-1])
    
    def update_weights(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.d_weights[i]
            self.biases[i] -= lr * self.d_biases[i]
    
    def train(self, X_train, y_train, epochs, lr, batch_size=32, verbose=1):
        history = []
        batch_size = min(batch_size, len(X_train))
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
                self.update_weights(lr)
            
            loss = self.loss(y_train, self.forward(X_train))
            history.append(loss)
            
            if verbose and (epoch+1) % (epochs//10) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        
        return history
    
    def predict(self, X):
        output = self.forward(X)
        
        if output.shape[1] == 1:  
            return (output > 0.5).astype(int)
        else: 
            return np.argmax(output, axis=1)

    def predict_proba(self, X):
        return self.forward(X)
    
    def visualize_network(self):
        G = nx.DiGraph()
        layer_nodes = []
        bias_nodes = []

        for i, size in enumerate(self.layers):
            nodes = [(i, j) for j in range(size)]
            layer_nodes.append(nodes)
            G.add_nodes_from(nodes, layer=i)
            
            if i > 0: 
                bias_node = (i - 1, 'b')
                bias_nodes.append(bias_node)
                G.add_node(bias_node, layer=i)

        edge_labels = {}
        node_labels = {}

        for i in range(len(self.layers) - 1):
            for j, a in enumerate(layer_nodes[i]):     
                for k, b in enumerate(layer_nodes[i+1]):
                    weight = self.weights[i][j, k]
                    G.add_edge(a, b, weight=weight)
                    edge_labels[(a, b)] = f"{weight:.6f}"
            
            for k, b in enumerate(layer_nodes[i+1]):
                bias = self.biases[i][0, k]
                bias_node = (i, 'b')
                G.add_edge(bias_node, b, weight=bias)
                edge_labels[(bias_node, b)] = f"{bias:.6f}"

        pos = {}
        for i, nodes in enumerate(layer_nodes):
            for j, node in enumerate(nodes):
                pos[node] = (i, -j)
        
        for i, bias_node in enumerate(bias_nodes):
            pos[bias_node] = (bias_node[0], -len(layer_nodes[bias_node[0]]) - 1)

        plt.figure(figsize=(10, 6))
        nx.multipartite_layout(G, subset_key="layer")
        nx.draw(G, pos, with_labels=True, node_size=1000, edge_color='gray', node_color='lightblue', font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="red", font_size=8)

        plt.title("Neural Network Structure with Weights & Biases")
        plt.show()
        
    def plot_weight_distribution(self, layers):
        plt.figure(figsize=(8,6))
        for layer in layers:
            plt.hist(self.weights[layer].flatten(), bins=30, alpha=0.7, label=f'Layer {layer}')
        plt.legend()
        plt.title("Weight Distribution")
        plt.xlabel("Weight Values")
        plt.ylabel("Frequency")
        plt.show()
    
    def plot_gradient_distribution(self, layers):
        plt.figure(figsize=(8,6))
        for layer in layers:
            plt.hist(self.d_weights[layer].flatten(), bins=30, alpha=0.7, label=f'Layer {layer}')
        plt.legend()
        plt.title("Gradient Distribution")
        plt.xlabel("Gradient Values")
        plt.ylabel("Frequency")
        plt.show()
    
    def save_model(self, filename):
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layers': self.layers,
            'activations': [act.__name__ for act in self.activations],
            'loss': self.loss.__name__
        }
        np.savez(filename, **model_data)
    
    @staticmethod
    def load_model(filename):
        data = np.load(filename, allow_pickle=True)
        ffnn = FFNN(
            layers=data['layers'], 
            activations=data['activations'], 
            loss=data['loss']
        )
        ffnn.weights = list(data['weights'])
        ffnn.biases = list(data['biases'])
        return ffnn

if __name__ == "__main__":
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]]) 

    ffnn = FFNN(
        layers=[2, 2, 2, 1], 
        activations=["relu", "sigmoid", "sigmoid"], 
        loss="mse",
        init_method="uniform", lower=-0.5, upper=0.5, seed=42
    )

    history = ffnn.train(X_train, y_train, epochs=5000, lr=0.1, batch_size=2, verbose=1)

    y_pred = ffnn.predict(X_train)
    print(y_pred)

    ffnn.visualize_network()
    ffnn.plot_weight_distribution([0, 1])
    ffnn.plot_gradient_distribution([0, 1])