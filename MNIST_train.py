import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
from MNIST_Network import *
import random
import pickle



def get_data(test_or_train, max_amount):
    # Returns a the first max_amount entries (or all of the data) from the test_or_train MNIST data.
    # - allowed_values is a list containing the allowed integers (int datatype)
    all_data = []
    with open("mnist_" + test_or_train + ".csv", "r") as csv_file:
        i = 0
        for data in csv.reader(csv_file):
            label = data[0]
            # Desired output
            desired_output = np.zeros((10))
            desired_output[int(label)] = 1
            # Pixels
            pixels = data[1:]
            input = np.array(pixels, "uint8")
            all_data.append((desired_output, input))
            i += 1
            if i == max_amount:
                break
        return all_data


def train_through_data(network, data, sample_size, learning_rate, test_data):
    # Divides the data into samples in the obvious order, and then trains on the samples. sample_cap is the maximum
    # number of samples that can be made
    number_of_samples = int(len(data) / sample_size)
    for i in range(number_of_samples):
        start_index = i * sample_size
        sample = data[start_index : start_index + sample_size]
        network.train(sample)
        if i % int(number_of_samples / 10) == 0:
            print('Current epoch: ', i, '/', number_of_samples)

def show_error(network, data, sample_size):
    # Prints the average loss function and accuracy
    print('Loss: ', network.get_average_error(data[0:sample_size]), 'Accuracy: ', str(100 * network.get_accuracy(data[0:sample_size])) + '%')

def train_network(network, data, sample_size, epochs, learning_rate, test_data):
    # Trains through the data epochs number of times, shuffling each time, and shows the error on the test data after
    # each epoch
    i = 0
    while i < epochs:
        print('epochs: ', i)
        i = i + 1
        train_through_data(network, data, sample_size, learning_rate, test_data)
        np.random.shuffle(data)
        show_error(network, test_data, len(test_data))
        
def save_net(net, filename):
    # Saves net to a file
    file = open(filename, 'wb')
    pickle.dump(net, file)













