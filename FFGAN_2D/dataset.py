import numpy as np
import os

class dataset():
	def __init__(self, kind, N=1000):
		self.N = N
		self.kind = kind
		self.data = self.get_data()
		self.save_data()


	def get_data(self):
		x = None
		N = self.N

		# Line Dataset (x*sin(x) + noise)
		if self.kind=='line':
			N = 2000
			t = np.linspace(-1,1,N)
			n = np.random.normal(loc=0, scale=0.1, size=N)
			X = t**2*np.sin(t) + n

			X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
			X = X_std * 2 -1

			x = np.concatenate([t.reshape(N,1), X.reshape(N,1)], axis=1)

		# Gaussian Lattice
		if self.kind=='gaussian_lattice':
			means = []
			covs = []
			for i in range(-2,2):
				for j in range(-2,2):
					means.append([i,j])
					covs.append(np.diag([0.01,0.01]))

			# Sample Points
			n = int(N/16)
			x = np.zeros((len(means)*n, 2))
			for i in range(len(means)):
				x[i*n:(i+1)*n,:] = np.random.multivariate_normal(means[i], covs[i], size=n)

			x_std = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
			x = x_std * 2 -1


		# Gaussian Circle
		if self.kind=='circle':
			x = np.zeros((N,2))
			theta = np.linspace(0,2*np.pi, N)
			R = 0.7
			for i,angle in enumerate(theta):
				x[i,:] = [R*np.cos(angle), R*np.sin(angle)] + np.random.normal(0, 0.08, size=[1,2])

			x = x.astype(np.float32)
		return x

	def save_data(self):
		data_out_dir = self.kind
		if not os.path.exists(data_out_dir):
			os.makedirs(data_out_dir)

		np.save(data_out_dir+'/data.npy', self.data)
		return