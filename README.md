This is a simple 2-layer neural network trained to identify a digit from a 28x28 pixel image.

The network consists of 784 = 28 * 28 input neurons (containing the grayscale value of each pixel), a hidden layer of 20 neurons, and a final output layer of 10 neurons (each neuron corresponds to a digit from 0 to 9).

The network only uses Sigmoid activator functions.

The backpropogation part of the code computes the gradient of the error function for the current example.
