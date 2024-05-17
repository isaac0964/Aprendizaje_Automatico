"""
A Librrary to implement a Fully Connected Neural Network
"""

import numpy as np
import pickle
from mnist_loader import load_mnist

# Activation Funcs
def linear(z, derivative=False):
    a = z 
    if derivative:
        da = 1
        return a, da
    return a

def sigmoid(z, derivative=False):
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = a * (1 - a)
        return a, da
    return a

def tanh(z, derivative=False):
    a = np.tanh(z)
    if derivative:
        da = (1 - a) * (1 + a)
        return a, da
    return a

def relu(z, derivative=False):
    a = z * (z >= 0)
    if derivative:
        da = np.array(z >= 0, dtype=float)
        return a, da
    return a

def softmax(z, derivative=False):
    c = np.max(z, axis=0)  # Termino para estabilizar
    e = np.exp(z - c)
    a = e / np.sum(e, axis=0)
    if derivative:
        da = np.ones(z.shape, dtype=float)
        return a, da
    return a

def load_pretrained_net(path:str="wb.txt"):
    """
    This funcitions loads a pretrained Fully Connected Network with the given weights
    input: path to pretrainded parameters
    ouput: trained FCN
    """
    data_file = open(path, "rb")
    ws, bs = pickle.load(data_file, encoding="latin1")
    data_file.close()
    layers_dims = [w.shape[1] for w in ws[1:]]
    layers_dims.append(ws[-1].shape[0])
    net = FCN(layers_dims)
    net.w = ws
    net.b = bs
    return net

class FCN():
    def __init__(self, layers_dims, hidden_activation=relu, output_activation=softmax, learning_rate=0.1):
        
        # Instantiate Attributes
        self.L = len(layers_dims) - 1
        self.w = [None]
        self.b = [None]
        self.f = [None]
        self.layers = layers_dims
        self.eta = learning_rate
        
        # Weight Initialization
        for l in range(1, self.L+1):
            self.w.append(-1 + 2 * np.random.rand(self.layers[l], self.layers[l-1]))
            self.b.append(-1 + 2 * np.random.rand(self.layers[l], 1))
            
            if l == self.L:
                self.f.append(output_activation)
            else:
                self.f.append(hidden_activation)
                
    def predict(self, X):
        A = X.copy()
        for l in range(1, self.L+1):
            A = self.f[l](self.w[l] @ A + self.b[l])
        return A
    
    def get_activations_of_all_layers(self, input_a):
        activations = [input_a.reshape((input_a.size, 1))]
        for bias, weight, func in zip(self.b[1:], self.w[1:], self.f[1:]):
            last_a = activations[-1]
            new_a = func(np.dot(weight, last_a) + bias)
            new_a = new_a.reshape((new_a.size, 1))
            activations.append(new_a)
        return activations
    
    def fit(self, X, Y, epochs=100, batch_size=1):
        # Gradient Descent
        for _ in range(epochs):
            # Shuffle Data
            idx = np.random.permutation(X.shape[1])
            X = X[:, idx]
            Y = Y[:, idx]

            # Validate Batch Size
            if batch_size < 1 or batch_size > X.shape[1]:
                print(f"Please select a valid batch size, valid batch size should be in the range: [1, {X.shape[1]}]")
                break

            # Train for each batch
            for i in range(int(np.ceil(X.shape[1] / batch_size))):
                # Take batch from X
                # This does not raise an error becuase numpy only takes elements
                # Inside limits
                X_batch = X[:, i * batch_size: (i+1) * batch_size]
                Y_batch = Y[:, i * batch_size: (i+1) * batch_size]

                # Initialize Activations and its derivatives
                As = []
                dA = [None]
                lg = [None] * (self.L+1)

                # Forward Propagation
                A = X_batch.copy() 
                As.append(A)
                for l in range(1, self.L+1):
                    A, da = self.f[l](self.w[l] @ A + self.b[l], derivative=True)
                    As.append(A)
                    dA.append(da)

                # Backpropagation
                for l in range(self.L, 0, -1):
                    if l == self.L:
                        lg[l] = (Y_batch - As[l]) * dA[l] 
                    else:
                        lg[l] = (self.w[l+1].T @ lg[l+1]) * dA[l] 
                # Update Weights and bias
                for l in range(1, self.L+1):
                    self.w[l] += self.eta/batch_size * (lg[l] @ As[l-1].T)
                    self.b[l] += self.eta/batch_size * np.sum(lg[l], axis=1, keepdims=True)
    # Save net weights and biases
    def save(self, path="wb.txt"):
        """
        Function for saving weights and biases of net
        """
        data_file = open(path, mode="wb")
        pickle.dump((self.w, self.b), data_file)
        data_file.close()


