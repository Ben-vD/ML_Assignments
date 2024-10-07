import numpy as np

class Layer:

    def __init__(self, n_inputs, n_nodes, actFunc):

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes

        self.weights = np.random.normal(size = (n_nodes, n_inputs))
        self.biases = np.random.normal(size = (n_nodes))

        self.actFunc = actFunc

    def calcOutput(self, inputs):
        activation_vals = np.dot(self.weights, inputs) + self.biases
        return self.actFunc(activation_vals)
    
    def updateWB(self, wbArr):
        self.weights = wbArr[0:self.weights.size].reshape(self.weights.shape).copy()
        self.biases = wbArr[self.weights.size:].copy()

    def getWBSizes(self):
        return self.weights.size + self.biases.size
    
    def getWB(self):
        return self.weights, self.biases