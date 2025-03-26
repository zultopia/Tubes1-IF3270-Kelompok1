import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from FFNN.Activation import Activation
from FFNN.Loss import Loss
from FFNN.Initializer import Initializer

class FFNN:
    def __init__(self, layers, activations, loss, init_method='uniform', l1_lambda=0.0, l2_lambda=0.0, rms_norm=False, gamma=1.0, beta=0.0, **init_params):
        self.layers = layers
        self.activations = [getattr(Activation, act) for act in activations]
        self.d_activations = [getattr(Activation, 'd_' + act) for act in activations]
        self.loss = getattr(Loss, loss)
        self.d_loss = getattr(Loss, 'd_' + loss)

        self.rms_norm = rms_norm
        self.gamma = gamma
        self.beta = beta

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
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
    
    def rms_norm_layer(self, x):
        mean_square = np.mean(x**2, axis=-1, keepdims=True) 
        norm = x / np.sqrt(mean_square) 
        return self.gamma * norm + self.beta, norm

    def forward(self, X):
        self.a = [X]
        self.z = []
        self.norms = []
        
        for i in range(len(self.weights)):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            
            if self.rms_norm:
                z, norm = self.rms_norm_layer(z)
                self.norms.append(norm)
            
            a = self.activations[i](z)
            self.z.append(z)
            self.a.append(a)
        
        return self.a[-1]
    
    def backward(self, X, y):
        m = y.shape[0]

        if self.activations[-1] == Activation.softmax: 
            if self.loss.__name__ == "categorical_cross_entropy":
                dz = self.a[-1] - y  # Softmax + CCE 
            else:
                dL_ds = self.d_loss(y, self.a[-1]) 
                J = Activation.d_softmax(self.z[-1])  # Jacobian softmax
                
                dz = np.zeros_like(dL_ds)  
                for b in range(m):  
                    dz[b] = J[b] @ dL_ds[b]
        else:
            dz = self.d_loss(y, self.a[-1]) * self.d_activations[-1](self.z[-1])

        for i in reversed(range(len(self.weights))):
            # rms_norm
            if self.rms_norm and i < len(self.norms): 
                norm = self.norms[i] 
                d_norm = dz * self.gamma 
                d_mean_square = np.mean(d_norm * norm, axis=-1, keepdims=True) 
                
                dz = d_norm / np.sqrt(np.mean(self.z[i]**2, axis=-1, keepdims=True)) - \
                    (self.z[i] * d_mean_square) / (np.mean(self.z[i]**2, axis=-1, keepdims=True))**1.5

            
            self.d_weights[i] = self.a[i].T @ dz / m
            self.d_biases[i] = np.sum(dz, axis=0, keepdims=True) / m

            self.d_weights[i] += self.l1_lambda * np.sign(self.weights[i]) + self.l2_lambda * self.weights[i]
            
            # Clipping gradien
            max_norm = 1.0
            w_norm = np.linalg.norm(self.d_weights[i])
            if w_norm > max_norm:
                self.d_weights[i] *= max_norm / w_norm

            b_norm = np.linalg.norm(self.d_biases[i])
            if b_norm > max_norm:
                self.d_biases[i] *= max_norm / b_norm

            
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
            
            if verbose and (epochs >= 10) and (epoch+1) % max(1, epochs//10) == 0:
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