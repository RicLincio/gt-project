# Import the Packages
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from IPython.display import clear_output

class GAN(object):
	def __init__(self, z_dim=10, alpha=0.5, lr=0.0001, kind=None):
		self.z_dim = z_dim # Input noise dimension
		self.alpha = alpha # Parameter of the loss
		self.lr = lr
		self.G, self.D, self.GAN = self.get_models() # Models
		self.loss_G, self.loss_D = None, None
		self.kind = kind

	def loss_boost(self, y_true, y_pred):
		return -K.log(y_pred)

	def get_models(self):

		# Generator
		G = Sequential()
		G.add(Dense(3, input_dim=self.z_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
		G.add(LeakyReLU(0.2))
		G.add(Dense(5))
		G.add(LeakyReLU(0.2))
		G.add(Dense(10))
		G.add(LeakyReLU(0.2))
		G.add(Dense(2, activation='tanh'))
		opt_G = Adam(lr=self.lr, beta_1=0.5)
		G.compile(loss='binary_crossentropy', optimizer=opt_G)

		# Discriminator
		D = Sequential()
		D.add(Dense(10, input_dim=2, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
		D.add(LeakyReLU(0.2))
		D.add(Dropout(0.3))
		D.add(Dense(5))
		D.add(LeakyReLU(0.2))
		D.add(Dropout(0.3))
		D.add(Dense(2))
		D.add(LeakyReLU(0.2))
		D.add(Dropout(0.3))
		D.add(Dense(1, activation='sigmoid'))
		opt_D = Adam(lr=self.lr, beta_1=0.5)
		D.compile(loss='binary_crossentropy', optimizer=opt_D)

		# GAN Model
		D.trainable = False
		z = Input(shape=(self.z_dim,))
		x = G(z)
		GAN = Model(inputs=z, outputs=D(x))
		opt_GAN = Adam(lr=self.lr, beta_1=0.5)
		GAN.compile(loss=self.loss_boost, optimizer=opt_GAN)

		return G, D, GAN

	def save_models(self):
		alpha_dir = self.kind+'/alpha_'+str(self.alpha)
		if not os.path.exists(alpha_dir):
			os.makedirs(alpha_dir)

		self.G.save(alpha_dir+'/G.h5')
		self.D.save(alpha_dir+'/D.h5')
		self.GAN.save(alpha_dir+'/GAN.h5')
		return

	def save_losses(self):
		alpha_dir = self.kind+'/alpha_'+str(self.alpha)
		if not os.path.exists(alpha_dir):
			os.makedirs(alpha_dir)

		np.save(alpha_dir+'/loss_G.npy', self.loss_G)
		np.save(alpha_dir+'/loss_D.npy', self.loss_D)
		return

	def save_generated_pts(self, epoch, examples=100, seed=0):
		alpha_dir = self.kind+'/'+'alpha_'+str(self.alpha)
		if not os.path.exists(alpha_dir):
			os.makedirs(alpha_dir)
		pts_out_dir = alpha_dir+'/pts_out'
		if not os.path.exists(pts_out_dir):
			os.makedirs(pts_out_dir)

		np.random.seed(seed=seed)
		noise = np.random.normal(0, 1, size=[examples, self.z_dim])
		generated_pts = self.G.predict(noise)
		np.save(pts_out_dir+'/gan_generated_pts_epoch_%d.npy' % epoch, generated_pts)

		return

	def train(self, x_tr, epochs=20, batch_size=128, generate_every=5):
		N_tr = x_tr.shape[0]
		batch_real_size = int(batch_size * self.alpha)
		batch_fake_size = batch_size - batch_real_size
		n_batch = N_tr // batch_real_size
		self.loss_G = np.zeros((epochs*n_batch,))
		self.loss_D = np.zeros((epochs*n_batch,))

		self.save_generated_pts(0)
		for e in range(1, epochs+1):
			idx_batch_tr = np.arange(N_tr)
			np.random.shuffle(idx_batch_tr)

			for i in range(n_batch):

				# Get a random set of input noise and images
				noise = np.random.normal(0, 1, size=[batch_fake_size, self.z_dim])
				x_batch_real_tr = x_tr[idx_batch_tr[i*batch_real_size:(i+1)*batch_real_size],:]

				# Generate fake MNIST images
				x_batch_fake_tr = self.G.predict(noise)
				x_batch_tr = np.concatenate([x_batch_real_tr, x_batch_fake_tr])

				# Labels for generated and real data
				y_batch_tr = np.zeros(batch_size)
				y_batch_tr[:batch_real_size] = 0.9

				# Train discriminator
				self.D.trainable = True
				loss_D_i = self.D.train_on_batch(x_batch_tr, y_batch_tr)

				# Train generator
				noise = np.random.normal(0, 1, size=[batch_fake_size, self.z_dim])
				y_batch_tr_G = np.ones(batch_fake_size)
				self.D.trainable = False
				loss_G_i = self.GAN.train_on_batch(noise, y_batch_tr_G)

				# Store loss of most recent batch from this epoch
				self.loss_G[(e-1)*n_batch + i] = loss_G_i
				self.loss_D[(e-1)*n_batch + i] = loss_D_i
			
				if i % 5 == 0:
					print('Alpha : ',self.alpha , '|| Epoch :', e,'|| Mini-Batch :', '{0:.2f}'.format(100.*(i+1) / n_batch),'%', '|| D Loss : ', loss_D_i, 
						'|| G Loss : ', loss_G_i, end='\r')
			print('Alpha : ',self.alpha , '|| Epoch :', e,'|| Mini-Batch :', '{0:.2f}'.format(100.*(i+1) / n_batch),'%', '|| D Loss : ', loss_D_i, 
				'|| G Loss : ', loss_G_i)

			if (e % generate_every == 0):
				self.save_generated_pts(e)
		self.save_models()
		self.save_losses()

		return
