"""
This is a library to load MNIST data.
"""

import gzip
import pickle
import numpy as np

def one_hot(i):
    """
    This function creates a one-hot vecor for the digit ([0-9]) passed to the function
    """
    vec = np.zeros(10)
    vec[i] = 1
    return vec


def load_mnist(path:str="/Users/isaacg/Desktop/Maestría/Segundo_Semestre/Aprendizaje_Automatico/Datasets/mnist.pkl.gz"):
    """
    This functions returns MNIST data.
    Takes as input the path where mnist gz file is located.
    Returns a tuple containing ((X_train, y_train), (X_test, y_test))
    where X_train and X_test are (50000, 784) and (10000, 784) numpy.ndarrays, respectively,
    containing each input image as a 784-dimensional numpy.ndarray.
    y_train adn y_test are (50000, 10), (10000,) numpy.ndarrays, respectively,
    y_traincontanins the one-hot encoding for the correct label of each digit in X_train.
    y_test contains the corresponding classification of each instance in X_test, i.e. an integer.
    """

    file = gzip.open(path) 
    unpickler = pickle._Unpickler(file, encoding="latin1")
    data = unpickler.load()

    # Get only train and test data
    (X_train, y_train), _, (X_test, y_test) = data
    file.close()
    # One-hot encode y
    y_train = np.array([one_hot(y) for y in y_train])
    return (X_train, y_train), (X_test, y_test)