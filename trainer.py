from Network import Network
from Train import Train

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def map_data(num):
    return num / 255

if __name__ == "__main__":
    trainer = Train()
    trainer.train("mnist_train.csv", 200)
    trainer.save("weights1.txt")

    '''net = Network("weights.txt")
    train_data = np.array(pd.read_csv("mnist_test.csv"))
    test_input = train_data[:1, 1:].transpose()
    test_input = np.vectorize(map_data)(test_input)
    test_label = train_data[:1, 0].transpose()

    plt.gray()
    plt.imshow(test_input.reshape(28, 28), interpolation="nearest")
    plt.show()
    
    result = net.feed(test_input)
    result = np.argmax(result, axis=0)

    accuracy = np.mean(result == test_label)

    print(f"label: {test_label}")
    print(f"result: {result}")
    print(f"accuracy: {accuracy}")'''