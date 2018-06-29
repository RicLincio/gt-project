# Import Packages
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from FFGAN import GAN

# Load MNIST data
(x_tr, _), (x_te, _) = mnist.load_data()
x = np.concatenate([x_tr, x_te])
x = (x_tr.astype(np.float32) - 127.5)/127.5
x = x_tr.reshape(-1, 784)

# Model
alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
for a in alpha:
	model = GAN(alpha=a, lr=1e-4)
	model.train(x, epochs=100, batch_size=256, generate_every=5)


