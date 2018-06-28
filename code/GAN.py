# Improt the Packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from termcolor import colored

# Define the Gan Class
class GAN(object):
	def __init__(self, lr=1E-3, batch_size=128, z_dim=100, epochs=75, data=None, name=None, D_kind='FF', G_kind='FF'):
		self.lr = lr
		self.batch_size = batch_size
		self.z_dim = z_dim
		self.epochs = epochs
		self.data = data
		self.name = str(name)
		self.D_kind = D_kind
		self.G_kind = G_kind
		self.str_info = D_kind + '_' + G_kind + '_' + str(epochs) + '_' + str(lr) + '_' + str(batch_size) + '_' + str(name)

		self.z_in = tf.placeholder(tf.float32, shape=[None, z_dim])
		self.x_in = tf.placeholder(tf.float32, shape=[None, 784])

		# Construct Discriminator/ Generator graph ops
		self.g_sample, self.g_weights = self.generator(z=self.z_in)
		self.d_real, self.d_weights = self.discriminator(self.x_in)
		self.d_fake, _ = self.discriminator(self.g_sample, reuse=True)

		# Loss and optimization ops
		self.d_loss, self.g_loss = self.loss()
		#self.d_loss_te, self.g_loss_te = self.loss()
		self.d_train, self.g_train = self.optimize()

		# Initialize session to run ops in
		self._sess = tf.Session()
		self._sess.run(tf.global_variables_initializer())

		# TensorBoard
		self.setupTB()
		self.merged = tf.summary.merge_all()
		self.writer_tr = tf.summary.FileWriter('tensorboard/'+self.str_info+'/tr', self._sess.graph)
		self.writer_te = tf.summary.FileWriter('tensorboard/'+self.str_info+'/te', self._sess.graph)

	def discriminator(self, x, reuse=False):
		
		# Output of the Discriminator
		out = 0
		parameters = []

		# Convolutional Model
		if self.D_kind=='CNN':
			# => Current Shape = (batch_size, 784)

			# Reshape the Input
			x_image = tf.reshape(x, [-1, 28, 28, 1])
			# => Current Shape = (batch_size, 28, 28, 1)

			with tf.variable_scope("Discriminator_CNN", reuse=reuse):

				# Convolutional Layer 1
				d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
				d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0.1))
				d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
				d1 = tf.nn.leaky_relu(d1 + d_b1)
			    # => Current Shape = (batch_size, 28, 28, 32)
			    
			    # Pooling Layer 1
				d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
			    # => Current Shape = (batch_size, 14, 14, 32)
			    
			    # Convolutional Layer 2
				d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
				d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0.1))
				d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
				d2 = tf.nn.leaky_relu(d2 + d_b2) 
			    # => Current Shape = (batch_size, 14, 14, 64)
			    
			    # Pooling Layer 2
				d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			    # => Current Shape = (batch_size, 7, 7, 64)
			 
			    # Fully Connected Layer 1
				d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
				d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0.1))
				d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
				d3 = tf.matmul(d3, d_w3)
				d3 = tf.nn.leaky_relu(d3 + d_b3)
			    # => Current Shape = (batch_size, 1024)

			    # Fully Connected Layer 2
				d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
				d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0.1))
				d4 = tf.matmul(d3, d_w4) + d_b4
			    # => Current Shape = (batch_size, 1)

				out = d4
				parameters = [d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_w4, d_b4]

		# Deep Feed Forward Model
		if self.D_kind=='FF':
			with tf.variable_scope("Discriminator_FF", reuse=reuse):
				W1 = tf.get_variable("W1", shape=[784, 128], initializer=tf.contrib.layers.xavier_initializer())
				b1 = tf.get_variable("b1", shape=[128], initializer=tf.constant_initializer(0.0))
				d_h1 = tf.nn.elu(tf.add(tf.matmul(x, W1), b1))

				W2 = tf.get_variable("W2", shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
				b2 = tf.get_variable("b2", shape=[1], initializer=tf.constant_initializer(0.0))
				d_h2 = tf.add(tf.matmul(d_h1, W2), b2)

				out = tf.nn.sigmoid(d_h2)
				parameters = [W1,b1,W2,b2]

		return out, parameters

	def generator(self, z):
		out = None
		parameters = []

		# Deconvolutional Model
		if self.G_kind=='DCNN':
			with tf.variable_scope("Generator_DCNN", reuse=False):

			    # Fully Connected Layer 1 Deconvolution
			    g_w1 = tf.get_variable('g_w1', [self.z_dim, 3136], dtype=tf.float32, 
			                           initializer=tf.truncated_normal_initializer(stddev=0.02))
			    g_b1 = tf.get_variable('g_b1', [3136], 
			                           initializer=tf.truncated_normal_initializer(stddev=0.02))
			    g1 = tf.matmul(z, g_w1) + g_b1
			    g1 = tf.reshape(g1, [-1, 56, 56, 1])
			    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
			    g1 = tf.nn.relu(g1)
			    # => Current Shape = (batch_size, 56, 56, 1)

			    # Deconvolutional Layer 1
			    g_w2 = tf.get_variable('g_w2', [3, 3, 1, self.z_dim/2], dtype=tf.float32, 
			                           initializer=tf.truncated_normal_initializer(stddev=0.02))
			    g_b2 = tf.get_variable('g_b2', [self.z_dim/2], 
			                           initializer=tf.truncated_normal_initializer(stddev=0.02))
			    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
			    g2 = g2 + g_b2
			    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
			    g2 = tf.nn.relu(g2)
			    g2 = tf.image.resize_images(g2, [56, 56])
			    # => Current Shape = (batch_size, 56, 56, 50)

			    # Deconvolutional Layer 2
			    g_w3 = tf.get_variable('g_w3', [3, 3, self.z_dim/2, self.z_dim/4], dtype=tf.float32, 
			                           initializer=tf.truncated_normal_initializer(stddev=0.02))
			    g_b3 = tf.get_variable('g_b3', [self.z_dim/4], 
			                           initializer=tf.truncated_normal_initializer(stddev=0.02))
			    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
			    g3 = g3 + g_b3
			    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
			    g3 = tf.nn.relu(g3)
			    g3 = tf.image.resize_images(g3, [56, 56])
			    # => Current Shape = (batch_size, 56, 56, 25)

			    # Deconvolutional Layer 3
			    g_w4 = tf.get_variable('g_w4', [1, 1, self.z_dim/4, 1], dtype=tf.float32, 
			                           initializer=tf.truncated_normal_initializer(stddev=0.02))
			    g_b4 = tf.get_variable('g_b4', [1], 
			                           initializer=tf.truncated_normal_initializer(stddev=0.02))
			    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
			    g4 = g4 + g_b4
			    g4 = tf.sigmoid(g4)
			    # => Current Shape = (batch_size, 28, 28, 1)

			    g4 = tf.reshape(g4, [-1, 784])
			    # => Current Shape = (batch_size, 784)

			    out = g4
			    parameters = [g_w1, g_b1, g_w2, g_b2, g_w3, g_b3, g_w4, g_b4]


		# Deep Feed Forward Model
		if self.G_kind=='FF':
			with tf.variable_scope("Generator_FF", reuse=False):
				W1 = tf.get_variable("W1", shape=[self.z_dim, 128], initializer=tf.contrib.layers.xavier_initializer())
				b1 = tf.get_variable("b1", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
				g_h1 = tf.nn.elu(tf.add(tf.matmul(z, W1), b1))

				W2 = tf.get_variable("W2", shape=[128, 784], initializer=tf.contrib.layers.xavier_initializer())
				b2 = tf.get_variable("b2", shape=[784], initializer=tf.contrib.layers.xavier_initializer())
				g_h2 = tf.add(tf.matmul(g_h1, W2), b2)

				out = tf.nn.sigmoid(g_h2)
				parameters = [W1,b1,W2,b2]

		return out, parameters

	def loss(self):
		discriminator_loss = -tf.reduce_mean(tf.log(self.d_real) + tf.log(1. - self.d_fake))
		generator_loss = -tf.reduce_mean(tf.log(self.d_fake))
		return discriminator_loss, generator_loss

	def optimize(self):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		d_train = optimizer.minimize(self.d_loss, var_list=self.d_weights)
		g_train = optimizer.minimize(self.g_loss, var_list=self.g_weights)
		return d_train, g_train

	def sample_z(self, num_samples, seed=None):
		np.random.seed(seed=seed)
		return np.random.uniform(-1., 1., size=[num_samples, self.z_dim])

	def train_discriminator(self, x_in, write_summary=False, iteration=None):
		z_sample = self.sample_z(self.batch_size)
		#fetches = [self.d_train, self.d_loss]
		#_, d_loss = self._sess.run(fetches, feed_dict={self.x_in: x_in, self.z_in:z_sample})
		if write_summary:
			_, d_loss, summary = self._sess.run([self.d_train, self.d_loss, self.merged], feed_dict={self.x_in: x_in, self.z_in:z_sample})
			self.writer_tr.add_summary(summary, iteration)
		else:
			_, d_loss = self._sess.run([self.d_train, self.d_loss], feed_dict={self.x_in: x_in, self.z_in:z_sample})

		return d_loss

	def train_generator(self, write_summary=False, iteration=None):
		z_sample = self.sample_z(self.batch_size)
		#fetches = [self.g_train, self.g_loss]
		#_, g_loss = self._sess.run(fetches, feed_dict={self.z_in: z_sample})
		if write_summary:
			_, g_loss, summary = self._sess.run([self.g_train, self.g_loss, self.merged], feed_dict={self.z_in: z_sample})
			self.writer_tr.add_summary(summary, iteration)
		else:
			_, g_loss = self._sess.run([self.g_train, self.g_loss], feed_dict={self.z_in: z_sample})
		return g_loss

	def write_discriminator_loss(self, x_in, iteration=None, mode='train'):
		z_sample = self.sample_z(self.batch_size)
		d_loss, summary = self._sess.run([self.d_loss, self.merged], feed_dict={self.x_in: x_in, self.z_in:z_sample})
		if mode=='train':
			self.writer_tr.add_summary(summary, iteration)
		if mode=='test':
			self.writer_te.add_summary(summary, iteration)
		return

	def write_generator_loss(self, iteration=None, mode='train'):
		z_sample = self.sample_z(self.batch_size)
		g_loss, summary = self._sess.run([self.g_loss, self.merged], feed_dict={self.z_in: z_sample})
		if mode=='train':
			self.writer_tr.add_summary(summary, iteration)
		if mode=='test':
			self.writer_te.add_summary(summary, iteration)
		return

	def sample_g(self, num_samples, seed=None):
		z_sample = self.sample_z(num_samples=num_samples, seed=seed)
		return self._sess.run(self.g_sample, feed_dict={self.z_in: z_sample})

	def save_image(self, images, plot_index):
		# Directory Stuff
		script_dir = os.path.dirname('out/')
		if not os.path.isdir(script_dir):
		    os.makedirs(script_dir)
		results_dir = os.path.join(script_dir, self.str_info)
		if not os.path.isdir(results_dir):
		    os.makedirs(results_dir)
		out_dir = 'out/'+self.str_info

		# Save Image
		fig = plt.figure(figsize=(4, 4))
		for i in range(len(images)):
			plt.subplot(4, 4, i+1)
			plt.imshow(images[i,:].reshape((28, 28)), cmap='gray')
			plt.axis('off')

		fig.savefig(out_dir+'/{}.png'.format(str(plot_index).zfill(3)), bbox_inches='tight')
		plt.close(fig)
		return

	def fit(self):
		plot_index = 0
		iteration = 0
		for epoch in range(self.epochs):
			count_batch = 0
			for batch in range(self.data.train.num_examples // self.batch_size):
				batch_x, _ = self.data.train.next_batch(self.batch_size)

				str_model_name = '\033[1m' + colored('Model Name', 'green')+'\033[0;0m'+' : '+ self.name
				str_model_kind = '\033[1m' + colored('D Kind', 'green')+'\033[0;0m'+' : '+ self.D_kind + ', ' + '\033[1m' + colored('G Kind', 'green')+'\033[0;0m'+' : '+ self.G_kind
				str_lr = '\033[1m' + colored('lr', 'green')+'\033[0;0m'+' : '+ str(self.lr)
				str_epoch = '\033[1m'+ colored('Epoch', 'green') + '\033[0;0m'+' : '+str(epoch+1)+'/'+str(self.epochs)
				str_batch = '\033[1m'+ colored('Mini-Batch','green')+'\033[0;0m'+' : '+ '{0:.2f}'.format(100.*count_batch / self.data.train.num_examples * 100)+'%'
				str_line = '--------------------------------------------------\n'
				print(str_model_name + ' | '+str_model_kind + ' | '+ str_lr + ' | ' + str_epoch+' | '+str_batch, end='\r'*2, flush=True)

				if iteration % 10 == 0:
					batch_x_te, _ = self.data.test.next_batch(self.batch_size)
					# TensorBoard
					_ = self.train_discriminator(x_in=batch_x, write_summary=True, iteration=iteration)
					_ = self.train_generator()
					self.write_discriminator_loss(x_in=batch_x_te, iteration=iteration, mode='test')
				else:
					_ = self.train_discriminator(x_in=batch_x)
					_ = self.train_generator()

				if count_batch % 10 == 0:
					d_loss = self.train_discriminator(x_in=batch_x)
					g_loss = self.train_generator()
					str_loss = '\033[1m'+colored('Training D Loss', 'green')+'\033[0;0m : '+'{0:.6f}'.format(d_loss) + ' | '+'\033[1m'+colored('Training G Loss', 'green')+'\033[0;0m'+' : '+'{0:.6f}'.format(g_loss)
					print(str_model_name + ' | '+str_model_kind + ' | '+ str_lr + ' | ' + str_epoch+' | '+str_batch+' | '+str_loss, end='\r')
					#print(str_line, end='\r')
					#print("Epoch {} Discriminator Loss {} Generator loss {}".format(epoch + 1, d_loss, g_loss))
					gen_sample = self.sample_g(num_samples=16, seed=0)

					# Save Image
					# self.save_image(gen_sample, plot_index)
					# plot_index += 1

				count_batch += 1
				iteration += 1

			#if epoch % 10 == 0:
				#d_loss = self.train_discriminator(x_in=batch_x)
				#g_loss = self.train_generator()
				#print("Epoch {} Discriminator Loss {} Generator loss {}".format(epoch + 1, d_loss, g_loss))
				#gen_sample = self.sample_g(num_samples=16)

				# Save Image
				#self.save_image(gen_sample, plot_index)
				#plot_index += 1

			

	def setupTB(self):
		tf.summary.scalar('G Loss', self.g_loss)
		tf.summary.scalar('D Loss', self.d_loss)
		return
