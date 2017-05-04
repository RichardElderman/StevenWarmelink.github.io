import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('example.pgm')

print img.shape

sum = 0

a = []

for j in range(1,2300):

	sum = 0;
	for i in range (1,290):
		px = img[i,j][0]
		if px == 0:
			sum +=1
	a.append(sum);

plt.plot(a);
plt.show();
