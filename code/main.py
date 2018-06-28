# Improt the Packages
from GAN import GAN

# Load the Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define the Model
gan_feedforward = GAN(lr=1e-3, batch_size=128, z_dim=100, epochs=150, data=mnist, name=0, D_kind='FF', G_kind='FF')
#gan_convolutional = GAN(lr=1e-4, batch_size=128, z_dim=100, epochs=75, data=mnist, name=1, D_kind='CNN', G_kind='DCNN')

# Train
gan_feedforward.fit()
#gan_convolutional.fit()
