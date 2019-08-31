import MNIST_Network, MNIST_train, MNIST_test

# Initialise the net
net = MNIST_Network.MNIST_Net([784, 32, 32, 10], 1)

# Load the data
train_data = MNIST_train.get_data("train", 10000000)
test_data = MNIST_train.get_data("test", 100000000)

# Train the network
MNIST_train.train_network(net, train_data, 30, 5, 5, test_data)
MNIST_train.train_network(net, train_data, 50, 10, 0.5, test_data)

# Draw a table of predictions against correct answers
test_data = MNIST_test.get_data(100000000)
MNIST_test.make_table(net, test_data)
