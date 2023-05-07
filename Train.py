from Network import *

import pandas as pd
import numpy as np

class Train:
    def __init__(self):
        self.network = Network(self.randomize_weights(), self.randomize_biases())
        

        '''i = 0
        for row in train_data:
            print(", ".join(row))
            i += 1
            if i > 5:
                break'''
    
    def randomize_weights(self):
        weights = []

        for i in range(0, Settings.layers - 1):
            new_weight = np.random.rand(Settings.layout[i+1], Settings.layout[i]) - 0.5
            weights.append(new_weight)
        
        return weights
    
    def randomize_biases(self):
        biases = []

        for i in range(0, Settings.layers - 1):
            new_biases = np.random.rand(Settings.layout[i+1]) - 0.5
            biases.append(new_biases)
        
        return biases
    
    def save(self):
        self.network.save("weights.txt")

    # matrify this
    def get_cost(self, inputs: np.ndarray, labels: np.ndarray):
        result = self.network.feed(inputs, 0)
        target = np.zeros((labels.size, 10))
        target[np.arange(labels.size), labels] = 1
        cost = np.square(result - target.transpose())
        
        return np.sum(cost, axis=0)
    
    def get_cost_prime(self, inputs: np.ndarray, labels: np.ndarray):
        result = self.network.feed(inputs, 0)
        target = np.zeros((labels.size, 10))
        target[np.arange(labels.size), labels] = 1

        return result - target.transpose()
    
    def train(self, f_train_data: str):
        train_data = np.array(pd.read_csv(f_train_data))

        '''# calculate cost
        self.cost = self.get_cost(test_row[1:], test_row[0]) # input data, correct label
        print(f"cost: {self.cost}")
        #print(f"weighted sums: {self.network.weighted_sums}")
        print(f"neurons: {self.network.neurons[3]}")'''

        train_inputs = train_data[500:10000, 1:].transpose() # first 50 samples
        train_inputs = np.vectorize(map)(train_inputs) # map 0-255 to 0-1
        train_labels = train_data[500:10000, 0].transpose()

        test_inputs = train_data[:500, 1:].transpose()
        test_inputs = np.vectorize(map)(test_inputs)
        test_labels = train_data[:500, 0].transpose()

        # gradient descent
        for i in range(500):
            print(f"epoch: {i}")
            grad_w, grad_b = self.train_batch(train_inputs, train_labels)
            for j in range(self.network.layers - 1):
                self.network.weights[j] -= Settings.alpha * grad_w[j]
                self.network.biases[j] -= Settings.alpha * grad_b[j]
            
            if i % 5 == 0:
                cost = self.get_cost(test_inputs, test_labels)
                print(f"average cost: {cost.mean()}")
            

    def train_batch(self, inputs: np.ndarray, labels: np.ndarray): # input, label
        grad_weights = []
        grad_biases = []

        dz = self.get_cost_prime(inputs, labels) # forward propagate and get first dz
        self.back_propagate(self.network.layers - 1, dz, grad_weights, grad_biases)

        return grad_weights, grad_biases

    # recursion
    def back_propagate(self, layer: int, dz: np.ndarray, grad_w: list, grad_b: list):

        # print(f"Propagating to layer: {layer}")

        _, m = dz.shape # number of samples in batch
        a = self.network.neurons[layer-1] # previous layer activation
        w = self.network.weights[layer-1] # current layer weights

        dw = 1 / m * np.dot(dz, a.transpose())
        db = 1 / m * np.sum(dz, axis=1)

        # append dw and db to gradient
        grad_w.insert(0, dw)
        grad_b.insert(0, db)

        # return if last layer
        if layer == 1:
            return

        z = self.network.weighted_sums[layer-2] # previous layer weighted sums

        dz_next = np.dot(w.transpose(), dz)
        if self.network.activation == "sigmoid":
            dz_next *= np.vectorize(dsigmoid)(z)
        elif self.network.activation == "relu":
            dz_next *= np.vectorize(dReLU)(z)
        elif self.network.activation == "arctan":
            dz_next *= np.vectorize(darctan)(z)
        
        self.back_propagate(layer - 1, dz_next, grad_w, grad_b)


        '''if dcda.shape != self.network.neurons[layer].shape:
            print(f"Incorrect layer shape of {dcda.shape}")
            return
        
        if layer == self.network.layers - 1:
            dadz = 1 # if last layer, then dont apply derivative of activation function
        if (self.network.activation == "sigmoid"):
            dadz = np.vectorize(dsigmoid)(self.network.weighted_sums[layer-1])
        elif (self.network.activation == "arctan"):
            dadz = np.vectorize(darctan)(self.network.weighted_sums[layer-1])
        elif (self.network.activation == "relu"):
            dadz = np.vectorize(dReLU)(self.network.weighted_sums[layer-1])
        
        delta = dcda * dadz

        # change in weights
        dzdw = self.network.neurons[layer-1]
        dcdw = np.outer(delta, dzdw)
        self.grad_weights.insert(0, dcdw)
        print(f"dcdw: {dcdw}")

        # change in biases
        dzdb = 1
        dcdb = dzdb * delta
        self.grad_biases.insert(0, dcdb)

        # change to prev layer
        dzda = self.network.weights[layer-1].transpose()
        dcda_next = np.dot(dzda, delta)

        if (layer == 1):
            return
        
        print()

        self.back_propagate(layer - 1, dcda_next)'''

def map(num):
    return num / 255