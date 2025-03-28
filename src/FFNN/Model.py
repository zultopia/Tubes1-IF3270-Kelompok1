import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import dash
import dash_cytoscape as cyto
from dash import html
import pickle

from FFNN.Activation import Activation
from FFNN.Loss import Loss
from FFNN.Initializer import Initializer

class FFNN:
    def __init__(self, layers, activations, loss, init_method='uniform', l1_lambda=0.0, l2_lambda=0.0, rms_norm=False, gamma=1.0, **init_params):
        self.layers = layers
        self.activations = [getattr(Activation, act) for act in activations]
        self.d_activations = [getattr(Activation, 'd_' + act) for act in activations]
        self.loss = getattr(Loss, loss)
        self.d_loss = getattr(Loss, 'd_' + loss)

        self.rms_norm = rms_norm
        self.gamma = gamma
        self.epsilon = 1e-8

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
        norm = x / np.sqrt(mean_square + self.epsilon)
        return self.gamma * norm, norm


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
            # backprop RMSNorm
            if self.rms_norm and i < len(self.norms): 
                norm = self.norms[i]  
                mean_square = np.mean(self.z[i]**2, axis=-1, keepdims=True) 
                d_norm = dz * self.gamma  
                d_mean_square = np.mean(d_norm * norm, axis=-1, keepdims=True)  
                
                dz = d_norm / np.sqrt(mean_square + self.epsilon) - (self.z[i] * d_mean_square) / ((mean_square + self.epsilon) * np.sqrt(mean_square + self.epsilon))

            self.d_weights[i] = self.a[i].T @ dz / m
            self.d_biases[i] = np.sum(np.array(dz), axis=0, keepdims=True) / m
            
            # regularisasi
            self.d_weights[i] += self.l1_lambda * np.sign(self.weights[i]) + self.l2_lambda * self.weights[i]
            
            if i > 0:
                dz = dz @ self.weights[i].T * self.d_activations[i-1](self.z[i-1])
    
    def update_weights(self, lr):
        max_norm = 1.0
        for i in range(len(self.weights)):
            w_norm = np.linalg.norm(self.d_weights[i])
            if w_norm > max_norm:
                self.d_weights[i] *= max_norm / w_norm

            b_norm = np.linalg.norm(self.d_biases[i])
            if b_norm > max_norm:
                self.d_biases[i] *= max_norm / b_norm
            self.weights[i] -= lr * self.d_weights[i]
            self.biases[i] -= lr * self.d_biases[i]
    
    def train(self, X_train, y_train, X_test, y_test, epochs, lr, batch_size=200, verbose=1):
        training_loss = []
        validation_loss = []
        batch_size = min(batch_size, len(X_train))

        if isinstance(y_train, np.ndarray) is False:
            y_train = y_train.to_numpy()
        if isinstance(X_train, np.ndarray) is False:
            X_train = X_train.to_numpy()

        if X_test is not None and y_test is not None:
            if not isinstance(y_test, np.ndarray):
                y_test = y_test.to_numpy()
            if not isinstance(X_test, np.ndarray):
                X_test = X_test.to_numpy()
            if y_test.ndim == 1 or y_test.shape[1] == 1:
                encoder = OneHotEncoder(sparse_output=False)
                y_test = encoder.fit_transform(y_test.reshape(-1, 1))

        if y_train.ndim == 1 or y_train.shape[1] == 1:
            encoder = OneHotEncoder(sparse_output=False)
            y_train = encoder.fit_transform(y_train.reshape(-1, 1))

        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)
                self.update_weights(lr)

            t_loss = self.loss(y_train, self.forward(X_train))
            training_loss.append(t_loss)

            if (verbose):
                print(f"[Epoch ({epoch+1}|{epochs})] Training Loss: {t_loss}", end="")

                if X_test is not None and y_test is not None:
                    v_loss = self.loss(y_test, self.forward(X_test))
                    validation_loss.append(v_loss)
                    print(f" - Validation Loss: {v_loss}")

        if X_test is not None and y_test is not None:
            return training_loss, validation_loss
        
        return training_loss
    
    def predict(self, X):
        if isinstance(X, np.ndarray) is False:
            X = X.to_numpy()

        output = self.forward(X)
        
        if output.shape[1] == 1:  
            return (output > 0.5).astype(int)
        else: 
            return np.argmax(output, axis=1)

    def predict_proba(self, X):
        if isinstance(X, np.ndarray) is False:
            X = X.to_numpy()

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
        edges = []

        for i in range(len(self.layers) - 1):
            for j, a in enumerate(layer_nodes[i]):
                for k, b in enumerate(layer_nodes[i+1]):
                    weight = self.weights[i][j, k]
                    G.add_edge(a, b, weight=weight)
                    edges.append((a, b, weight))
                    edge_labels[(a, b)] = f"{weight:.6f}"

            for k, b in enumerate(layer_nodes[i+1]):
                bias = self.biases[i][0, k]
                bias_node = (i, 'b')
                G.add_edge(bias_node, b, weight=bias)
                edges.append((bias_node, b, bias))
                edge_labels[(bias_node, b)] = f"{bias:.6f}"

        pos = {}
        for i, nodes in enumerate(layer_nodes):
            for j, node in enumerate(nodes):
                pos[node] = (i, -j)
        
        for i, bias_node in enumerate(bias_nodes):
            pos[bias_node] = (bias_node[0], -len(layer_nodes[bias_node[0]]) - 1)
        
        cytoscape_elements = []
        for node in G.nodes():
            cytoscape_elements.append({"data": {"id": str(node), "label": str(node)}, "position": {"x": pos[node][0] * 150, "y": pos[node][1] * 100}})
        
        for edge in edges:
            cytoscape_elements.append({"data": {"source": str(edge[0]), "target": str(edge[1]), "label": f"{edge[2]:.6f}"}})
        
        app = dash.Dash(__name__)
        app.layout = html.Div([
            cyto.Cytoscape(
                id='cytoscape-network',
                elements=cytoscape_elements,
                style={'width': '100%', 'height': '600px'},
                layout={'name': 'preset'},
                stylesheet=[
                    {'selector': 'node', 'style': {'content': 'data(label)', 'background-color': '#007bff', 'color': 'white', 'text-halign': 'center', 'text-valign': 'center'}},
                    {'selector': 'edge', 'style': {'line-color': '#bbb', 'label': 'data(label)', 'font-size': '10px', 'color': 'black', 'text-background-color': 'white', 'text-background-opacity': 1, 'curve-style': 'bezier'}}
                ]
            )
        ])
        
        app.run_server(debug=True)
        
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
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print("Model saved successfully...")

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        ffnn = FFNN.__new__(FFNN)  
        ffnn.__dict__.update(data)
        
        print("Model loaded successfully...")
        return ffnn
    