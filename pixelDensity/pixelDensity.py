import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('example.pgm')

sum = 0

a = []

height, width, channels = img.shape

for j in range(1,width):

	sum = 0;
	for i in range (1,height):
		px = img[i,j][0]
		if px == 0:
			sum +=1
	a.append(sum);

plt.plot(a);
plt.show();
