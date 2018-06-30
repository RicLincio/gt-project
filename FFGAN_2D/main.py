# Import Packages
from dataset import dataset
from FFGAN import GAN
import numpy as np

# Build Dataset
kind = 'line' # can be ['line', 'circle']
x = dataset(kind, N=1000).data

# Model
for a in [0.9, 0.7, 0.5, 0.3, 0.1]:
	model = GAN(alpha=a, lr=2e-4, z_dim=10, kind=kind)
	model.train(x, epochs=3000, batch_size=128, generate_every=50)