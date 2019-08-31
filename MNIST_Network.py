import numpy as np


class MNIST_Net:
    def __init__(self, layer_sizes, learning_rate):
        # Get all layer sizes and store them in a variable called r
        r = layer_sizes
        self.r = r

        # Set learning rate
        self.learning_rate = learning_rate

        # Number of layers
        m = len(layer_sizes) - 1
        self.m = m


        # ----------- Initialise all the numpy arrays. Would probably be  more efficient to use the same loop for
        # ----------- multiple variables, but this is only done once, so I'm separating them for simplicity

        # Activations
        a = []
        for k in range(m + 1):
            a.append(np.zeros((1, r[k])))
        self.a = a

        # Outputs
        o = []
        for k in range(m + 1):
            o.append(np.zeros((1, r[k] + 1)))
            o[k][0, 0] = 1
        self.o = o

        # Weights (the zeroth row of each weight matrix represents the biases)
        w = []
        for k in range(1, m + 1):
            w.append(np.random.randn(r[k - 1] + 1, r[k]))
        self.w = w

        # Derivative of each error contribution by each weight
        E_d_del = []
        for d in range(1, r[m] + 1):
            E_d_del.append([])
            for k in range(1, m + 1):
                E_d_del[d-1].append(np.zeros((r[k-1] + 1, r[k])))
        self.E_d_del = E_d_del

        # Derivative of total error function
        E_del = []
        for k in range(1, m + 1):
            E_del.append(np.zeros((r[k - 1] + 1, r[k])))
        self.E_del = E_del

    def set_learning_rate(self, rate):
        self.learning_rate = rate

    # Activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_prime(self, x):
        x[x > 0] = 1
        x[x<=0] = 0
        return x

    # The actual activation functions we are currently using
    def g(self, x):
        # Hidden layers
        return self.sigmoid(x)

    def g_zero(self, x):
        # Output layer
        return self.sigmoid(x)

    def g_prime(self, x):
        return self.sigmoid_prime(x)

    def g_zero_prime(self, x):
        return self.sigmoid_prime(x)


    # Feed forward

    def feed_forward(self):
        m = self.m
        a = self.a
        o = self.o
        w = self.w
        r = self.r

        # Update output of the zeroth layer

        o[0][0, 1 : r[0] + 1] = self.g(a[0])

        # Hidden layer feedforward loop
        for k in range(1, m):
            a[k] = np.matmul(o[k-1], w[k-1])
            o[k][0, 1: r[k] + 1] = self.g(a[k])

        # Output layer
        a[m] = np.matmul(o[m-1], w[m-1])
        o[m][0, 1: r[m] + 1] = self.g_zero(a[m])

    # Backpropagation

    def feed_backward(self, desired_output):
        m = self.m
        a = self.a
        o = self.o
        r = self.r
        w = self.w
        E_d_del = self.E_d_del
        E_del = self.E_del
        error = []

        # Init error list
        for k in range(1,m+1):
            error.append(np.zeros((1,r[k])))

        # Output error
        error[m-1] = self.g_zero_prime(a[m]) * (o[m][0, 1:r[m]+1] - desired_output)

        # Backprop the errors
        for k in range(m-1, 0, -1):
            error[k-1] = self.g_prime(a[k]) * np.matmul(error[k], np.transpose(w[k][1:r[k]+1]))

        # Calculate E_del
        for k in range(1, m+1):
            E_del[k-1] = np.matmul(o[k-1].reshape(-1,1), error[k-1])

    def train(self, sample):
        # Sample is a list of tuples, where the first entry is the desired output of the network and second entry
        # is the input.
        m = self.m
        r = self.r
        a = self.a
        w = self.w
        o = self.o
        sample_size = len(sample)
        # Initialise the average gradient list of matrices
        average_gradient = []
        for k in range(1, m + 1):
            average_gradient.append(np.zeros((r[k - 1] + 1, r[k])))

        # Add contributions from the sample to the average gradient
        for input_index in range(sample_size):
            a[0] = sample[input_index][1].reshape(1,784)
            self.feed_forward()
            self.feed_backward(sample[input_index][0].reshape(1,10))
            for k in range(1, m+1):
                average_gradient[k-1] += 1 / sample_size * self.E_del[k-1]
        # Adjust the weights and biases
        for k in range(1, m+1):
            w[k-1] = w[k-1] - self.learning_rate * average_gradient[k-1]

    def get_average_error(self, sample):
        # Takes the same sample as self.train, and returns the mean value of the mean-squared error loss function.
        a = self.a
        r = self.r
        m = self.m
        o = self.o
        sample_size = len(sample)
        total_error = 0
        for i in range(sample_size):
            a[0] = sample[i][1]
            self.feed_forward()
            total_error += 0.5 * np.linalg.norm(sample[i][0] - o[m][0, 1:r[m]+1])**2
        return total_error / sample_size

    # -------------- Prediction methods
    def evaluate(self, input):
        # Returns the output of the network for a given input
        self.a[0] = input
        self.feed_forward()
        return self.o[self.m][0, 1:self.r[self.m]+1]

    def make_prediction(self, datum):
        # Predicts which digit is represented by an image. Takes a tuple containing the desired output ([0]) and input
        # ([1])
        output = self.evaluate(datum)
        return self.get_actual_value(output)

    def get_actual_value(self, array):
        # Takes a 1x10 np array
        max_val = 0
        max_index = 0
        for i in range(len(array)):
            if array[i] > max_val:
                max_val = array[i]
                max_index = i
        return max_index

    def get_accuracy(self, sample):
        # Takes a sample and returns the proportion of correct answers by the network from that sample.
        sample_size = len(sample)
        number_correct = 0
        for i in range(sample_size):
            prediction = self.make_prediction(sample[i][1])
            if prediction == self.get_actual_value(sample[i][0]):
                number_correct += 1
        return number_correct / sample_size











