import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
from MNIST_Network import *
import random
import pickle


test_data = []

def get_data(max_amount):
    with open("mnist_test.csv", "r") as csv_file:
        all_data = []
        i = 0
        for data in csv.reader(csv_file):
            # The first column is the label
            label = data[0]
            # The rest of them are the pixels
            pixels = data[1:]
            pixels = np.array(pixels, "uint8")
            pixels = pixels.reshape((28,28))
            all_data.append((label, pixels))
            i += 1
            if i == max_amount:
                return all_data
    return all_data


def plot(index, test_data):
    # Takes pixels as square arrays
    (label, pixels) = test_data[index]
    plt.title(label)
    plt.imshow(pixels, cmap='gray')
    plt.show()

def load_net(filename):
    fileObject = open(filename, 'rb')
    net = pickle.load(fileObject)
    return net

def plot_prediction(net, index, test_data):
    # Takes pixels as square arrays
    pixels = test_data[index][1]
    prediction = net.make_prediction(pixels.reshape(784))
    plt.title('Prediction: ' + str(prediction))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    
def make_table(net, test_data):
    # Takes pixels as square arrays
    cell_text = [[0 for i in range(10)] for j in range(10)]
    cell_colour = [[(1,1,1) for i in range(10)] for j in range(10)]
    labels = []
    col_widths = [0.05 for i in range(10)]
    for i in range(10):
        labels.append(str(i))
    for i in range(len(test_data)):
        prediction = net.make_prediction(test_data[i][1].reshape(784))
        actual_value = int(test_data[i][0])
        cell_text[prediction][actual_value] += 1
        cell_colour[prediction][actual_value] = (1, max(0, cell_colour[prediction][actual_value][1] - 0.02), max(0, cell_colour[prediction][actual_value][2] - 0.02))

    plt.table(cellColours=cell_colour, cellText = cell_text, colLabels=labels, rowLabels=labels, loc='center')
    plt.show()

