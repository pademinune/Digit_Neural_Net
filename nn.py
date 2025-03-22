from data import get_mnist
import numpy as np


images, labels = get_mnist()
W_1 = np.random.uniform(-0.5, 0.5, (20, 784))
W_2 = np.random.uniform(-0.5, 0.5, (10, 20))
B_1 = np.zeros((20, 1))
B_2 = np.zeros((10, 1))

learn_rate = 0.1
nr_correct = 0
epochs = 3
for epoch in range(epochs):
    for N_1, l in zip(images, labels):
        N_1.shape += (1,)
        l.shape += (1,)
        
        # forward propogation
        N_2_pre = W_1 @ N_1 + B_1
        N_2 = 1 / (1 + np.exp(-N_2_pre))

        N_3_pre = W_2 @ N_2 + B_2
        N_3 = 1 / (1 + np.exp(-N_3_pre))

        # comparing output
        if (np.argmax(N_3) == np.argmax(l)):
            nr_correct += 1

        # back propogation for this specific training example. It computes the gradients of the error function for this specific example
        grad_N_3 = 2 * (N_3 - l)
        delta_N_3 = grad_N_3 * N_3 * (1 - N_3)

        grad_W_2 = delta_N_3 @ np.transpose(N_2)
        grad_B_2 = delta_N_3

        grad_N_2 = np.transpose(W_2) @ delta_N_3
        delta_N_2 = grad_N_2 * N_2 * (1 - N_2)

        grad_W_1 = delta_N_2 @ np.transpose(N_1)
        grad_B_1 = delta_N_2
        
        # gradient descending 1 step
        W_2 += -learn_rate * grad_W_2
        B_2 += -learn_rate * grad_B_2
        W_1 += -learn_rate * grad_W_1
        B_1 += -learn_rate * grad_B_1

        

    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

