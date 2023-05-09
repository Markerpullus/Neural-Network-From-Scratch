# network layer settings

class Settings:
    layers = 4
    layout = [784, 16, 16, 10]
    activation = "relu" # relu, sigmoid, tanh
    alpha = 0.7 # learning rate
    decay = 0.25 # decay rate
    batches = 100