# Import Packages
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from FFGAN import GAN

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)

# Model
alpha = [0.5, 0.7, 0.9]
for a in alpha:
    model = GAN(alpha=a, lr=1e-4)
    model.train(X_train, epochs=100, batch_size=256, generate_every=5)


