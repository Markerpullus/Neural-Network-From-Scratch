# network layer settings

class Settings:
    layers = 4
    layout = [784, 16, 16, 10]
    activation = "relu" # relu, sigmoid, tanh
    alpha = 0.5 # learning rate 0-1
    decay = 0.3 # decay rate 0-1
    batches = 200