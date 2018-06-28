import numpy as np
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import os


# Read images
str_dir = 'images/'
frames = []
list_dir_sort = sorted(os.listdir(str_dir), key=str.lower)

for i, str_img in enumerate(list_dir_sort):
	if str_img[0]!='.':
		frames.append(misc.imread(str_dir+str_img)[:,:,0:3])

	if i < len(os.listdir(str_dir))-1:
		print('Load Frames : '+'{0:.2f}'.format(100.*(i+1)/len(os.listdir(str_dir)))+'%', end='\r')
	else:
		print('Load Frames : '+'{0:.2f}'.format(100.*(i+1)/len(os.listdir(str_dir)))+'%')

h, w, c = frames[0].shape
n_frames = len(frames)

# Print Info
print('N Frames : '+str(n_frames))
print('Shapes : '+str(frames[0].shape))

# # Video
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 60.0, (w, h))

for i in range(n_frames):
	out.write(frames[i])
	cv2.imshow('Frame', frames[i])

	print('Write Video : '+'{0:.2f}'.format(100.*(i+1)/n_frames)+'%', end='\r')

out.release()
cv2.destroyAllWindows()