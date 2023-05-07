from Network import Network
from Train import Train

import numpy as np
import pandas as pd

def map_data(num):
    return num / 255

if __name__ == "__main__":
    trainer = Train()
    trainer.train("mnist_train.csv")
    trainer.save()

    '''net = Network("weights.txt")
    train_data = np.array(pd.read_csv("mnist_test.csv"))
    test_input = train_data[0:100, 1:].transpose()
    test_input = np.vectorize(map_data)(test_input)
    test_label = train_data[0:100, 0].transpose()
    
    result = net.feed(test_input, 0)
    result = np.argmax(result, axis=0)

    accuracy = np.mean(result == test_label)

    print(f"label: {test_label}")
    print(f"result: {result}")
    print(f"accuracy: {accuracy}")'''