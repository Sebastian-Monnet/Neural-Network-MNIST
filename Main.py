import numpy as np
from Network import *

def train_networks(number_of_networks, inputs_train, outputs_train, inputs_test, outputs_test, layer_sizes):
    nets = []
    errors = []
    for trial in range(number_of_networks):
        net = Net(layer_sizes, 1)
        for i in range(100000):
            if i % 100 == 0:
                print(i, net.get_average_error(inputs_test, outputs_test))
            net.train(inputs_train, outputs_train)
        average_error = net.get_average_error(inputs_test, outputs_test)
        print(average_error)
        nets.append(net)
        errors.append(average_error)



inputs_train = []
inputs_test = []
desired_outputs = []

for i in range(100):
    inputs_train.append(np.random.rand(1, 2))
    inputs_test.append(np.random.rand(1, 2))
    outputs_train = inputs_train
    outputs_test = inputs_test

train_networks(1, inputs_train, outputs_train, inputs_test, outputs_test, [2, 2])