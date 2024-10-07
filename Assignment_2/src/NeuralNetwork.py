import numpy as np
import Layer
import PSO
import UtilFunctions as uf
from functools import partial
import cPSO
import dPSO
import mPSO
import ActivationFunctions as af
from sklearn.metrics import accuracy_score
import crPSO

class NeuralNetwork:

    def __init__(self, n_features,  nodesArr, classification, l, actFuncs):

        function_map = {
            "linear": af.linear,
            "step": af.step,
            "ramp": af.ramp,
            "sigmoid": af.sigmoid,
            "hyperbolicTangent": af.hyperbolicTangent,
            "gaussian": af.gaussian,
            "relu": af.relu
        }
        functions_for_layers = [function_map[name] for name in actFuncs]

        self.layers = np.empty(len(nodesArr), dtype = object)
        self.nodesArr = nodesArr
        self.l = l

        self.classification = classification

        self.layers[0] = Layer.Layer(n_features, nodesArr[0], functions_for_layers[0])
        for i in range(1, len(nodesArr)):
            self.layers[i] = Layer.Layer(nodesArr[i - 1], nodesArr[i], functions_for_layers[i])

    def getLayerCount(self):
        return len(self.nodesArr)
    
    def getLayerSizes(self):
        return self.nodesArr

    def forwardPropogate(self, X):

        outputs = np.zeros((len(X), self.nodesArr[-1]))
        for i in range(0, len(X)):
            output = self.layers[0].calcOutput(X[i])
            for l in range(1, len(self.layers)):
                output = self.layers[l].calcOutput(output)
            outputs[i] = output

        return outputs
    
    def mseCost(self, X, y):

        y_pred = self.forwardPropogate(X)
        WD = self.weightDecay()
        if (self.classification):
            return (np.sum(np.sum(np.square(y - y_pred), axis = 1)) / X.size) + self.l * (WD / (WD + 1))
        return (np.sum(np.sum(np.square(y - y_pred))) / X.size) + self.l * (WD / (WD + 1))
    
    def mseCostWB(self, X, y, wbArr):
        self.updateWB(wbArr)
        return self.mseCost(X, y)

    def updateWB(self, wbArr):
        sIdx = 0
        for layer in self.layers:
            eIdx = sIdx + layer.getWBSizes()
            layer.updateWB(wbArr[sIdx:eIdx])
            sIdx = eIdx


    def weightDecay(self):
        w = 0
        for layer in self.layers:
            w += np.sum(np.square(layer.getWB()[0]))
            w += np.sum(np.square(layer.getWB()[1]))
        return w

    def getWBSizes(self):
        t = []
        for layer in self.layers:
            t.append(layer.getWBSizes())
        return np.array(t)

    def printWB(self):
        for layer in self.layers:
            print(layer.getWB())

    def fit(self, X, y, fitMethod):

        partial_cost_func = partial(self.mseCostWB, X, y)
        dim = np.sum(self.getWBSizes())

        c = 1.49618
        w = 0.729844

        fitness_arr = None
        diversity_arr = None

        if (fitMethod == 0):
            pso = PSO.PSO(10, 250, -1, 1, dim, partial_cost_func, c, c, w)
            newWeights, fitness_arr, diversity_arr = pso.run()
            self.updateWB(newWeights)
        elif (fitMethod == 1):
            cpso = cPSO.cPSO(10, 250, -1, 1, dim, partial_cost_func, c, c, w)
            newWeights, fitness_arr, diversity_arr = cpso.run()
            self.updateWB(newWeights)
        elif (fitMethod == 2):
            crpso = crPSO.crPSO(10, 250, -1, 1, dim, partial_cost_func, c, c, w)
            newWeights, fitness_arr, diversity_arr = crpso.run()
            self.updateWB(newWeights)
        elif (fitMethod == 3):
            dpso = dPSO.dPSO(10, 250, -1, 1, dim, partial_cost_func, c, c, w, 2)
            newWeights, fitness_arr, diversity_arr = dpso.run()
            self.updateWB(newWeights)
        elif (fitMethod == 4):
            mpso = mPSO.mPSO(10, 250, -1, 1, dim, partial_cost_func, c, c, w, 2)
            newWeights, fitness_arr, diversity_arr = mpso.run()
            self.updateWB(newWeights)

        return fitness_arr, diversity_arr

    def predict(self, X):
        y_pred = self.forwardPropogate(X)
        if(self.classification):
            return np.argmax(y_pred, axis = 1)
        return y_pred
    
    def score(self, X, y):
        if (self.classification):
            return accuracy_score(y, self.predict(X))
        return self.mseCost(X, y)
        
    def resetWB(self):
        dim = np.sum(self.getWBSizes())
        self.updateWB(np.random.normal(size = dim))