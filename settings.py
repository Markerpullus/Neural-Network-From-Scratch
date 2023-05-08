# network layer settings

class Settings:
    layers = 4
    layout = [784, 16, 16, 10]
    activation = "relu" # relu, sigmoid, tanh
    alpha = 0.05 # learning rate
    batches = 100