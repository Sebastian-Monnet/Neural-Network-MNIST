import numpy as np

class Net:
    def __init__(self, layer_sizes, learning_rate):
        # Takes a list of layer sizes as an input
        # Rename the layer sizes to comply with notation
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
            w.append(1 - 2 * np.random.rand(r[k - 1] + 1, r[k]))
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

    # Feed backward

    def feed_backward(self, target):
        m = self.m
        a = self.a
        #error = self.error
        o = self.o
        r = self.r
        w = self.w
        E_d_del = self.E_d_del
        E_del = self.E_del
        error = []
        for k in range(0,m+1):
            error.append(np.zeros((1,r[k])))


        # Set E_del to zero
        for k in range(1, m+1):
            E_del[k-1] = np.zeros((r[k-1] + 1, r[k]))

        # Calculate E_del
        for d in range(1, r[m]+1):
            # Final layer error terms
            error[m] = self.g_zero_prime(a[m]) * (o[m][0, d] - target[0,d-1])
            E_d_del[d - 1][m - 1] = np.matmul(o[m-1].reshape(-1, 1), error[m])
            E_del[m - 1] = E_del[m - 1] + E_d_del[d - 1][m - 1]
            # Hidden layer error terms
            for k in range(m-1, 0, -1):
                error[k] = self.g_prime(a[k]) * np.matmul(error[k+1], np.transpose(w[k][1:r[k] + 1]))
                # Calculate partial derivatives of E_d
                E_d_del[d - 1][k - 1] = np.matmul(o[k - 1].reshape(-1, 1), error[k])
                E_del[k-1] = E_del[k-1] + E_d_del[d-1][k-1]

    def train(self, inputs, desired_outputs):
        m = self.m
        r = self.r
        a = self.a
        w = self.w
        o = self.o
        sample_size = len(inputs)
        # Initialise the average gradient list of matrices
        average_gradient = []
        for k in range(1, m + 1):
            average_gradient.append(np.zeros((r[k - 1] + 1, r[k])))

        # Add contributions from the sample to the average gradient
        for input_index in range(sample_size):
            a[0] = inputs[input_index]
            self.feed_forward()
            self.feed_backward(desired_outputs[input_index])
            for k in range(1, m+1):
                average_gradient[k-1] += 1 / sample_size * self.E_del[k-1]
        # Adjust the weights and biases
        for k in range(1, m+1):
            w[k-1] -= self.learning_rate * average_gradient[k-1]

    def get_average_error(self, inputs, desired_outputs):
        a = self.a
        r = self.r
        m = self.m
        o = self.o
        sample_size = len(inputs)
        total_error = 0
        for i in range(sample_size):
            a[0] = inputs[i]
            self.feed_forward()
            total_error += 0.5 * np.linalg.norm(desired_outputs[i] - o[m][0, 1:r[m]+1])**2
        return total_error / sample_size








