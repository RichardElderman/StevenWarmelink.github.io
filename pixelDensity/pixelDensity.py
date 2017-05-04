import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('example2.pgm')

sum = 0



vertical_pixel_density = []

height, width, channels = img.shape

for j in range(1,width):

	sum = 0;
	for i in range (1,height):
		px = img[i,j][0]
		if px == 0:
			sum +=1
	vertical_pixel_density.append(sum);

seperators = []

prev_loc = 0

for i in range (1,len(vertical_pixel_density)):
	if not ((i - prev_loc) < 75):
		if vertical_pixel_density[i] < 4:
			prev_loc = i
			seperators.append(i)

print seperators

plt.subplot(2,1,1)
plt.plot(vertical_pixel_density);
plt.title('Vertical pixel density')

for i in range(0,len(seperators)):
	x = seperators[i]
	plt.plot([x,x],[0,height])


horizontal_pixel_density = []



for j in range(1,height):

	sum = 0;
	for i in range (1,width):
		px = img[j,i][0]
		if px == 0:
			sum +=1
	horizontal_pixel_density.append(sum);


plt.subplot(2,1,2)
plt.plot(horizontal_pixel_density);
plt.title('Horizontal pixel density')
plt.show();

for i in range(0,len(seperators)):
	for j in range(0,height-1):
		xval = seperators[i]
		img[j][xval] = [0,0,0]

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

