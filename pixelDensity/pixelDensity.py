import cv2
import numpy as np
from matplotlib import pyplot as plt

def rotateImage(img):
	img = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_CONSTANT,value=[255,255,255])
	
	height, width, channels = img.shape

	M = cv2.getRotationMatrix2D((width/2,height/2),1,1)
	img = cv2.warpAffine(img,M,(width,height))

	height, width, channels = img.shape

	img = img[100:height-101,100:width-101]

	return np.asarray(img)

def calcHorPixelDensity(img):
	vertical_pixel_density = []

	height, width, channels = img.shape
	for j in range(1,width):

		sum = 0;
		for i in range (1,height):
			px = img[i,j][0]
			if px == 0:
				sum +=1
		vertical_pixel_density.append(sum);

	return vertical_pixel_density

def calcVerPixelDensity(img):
	horizontal_pixel_density = []

	height, width, channels = img.shape
	for j in range(1,height):

		sum = 0;
		for i in range (1,width):
			px = img[j,i][0]
			if px == 0:
				sum +=1
		horizontal_pixel_density.append(sum);

	return horizontal_pixel_density

def CalcSeperators(vertical_pixel_density,minBoxWidth,threshold):
	seperators = []

	prev_loc = 0

	for i in range (0,len(vertical_pixel_density)):
		if not ((i - prev_loc) < minBoxWidth):
			if vertical_pixel_density[i] < threshold:
				prev_loc = i
				seperators.append(i)

	return seperators

def drawSeperators(img, seperators):
	height, width, channels = img.shape

	for i in range(0,len(seperators)):
		for j in range(0,height-1):
			xval = seperators[i]
			img[j][xval] = [0,0,0]

	cv2.imshow('image',img)
	cv2.waitKey(0)


def drawDensities(h_pixel_density, v_pixel_density):
	plt.subplot(2,1,1)
	plt.plot(v_pixel_density);
	plt.title('Vertical pixel density')

	plt.subplot(2,1,2)
	plt.plot(h_pixel_density);
	plt.title('Horizontal pixel density')
	plt.show();

"""
minwidth = []
print "vertical mean: ", v_mean 
print "vertical stdv: ", v_stdd
for i in range(0,len(vertical_pixel_density)):
	val = vertical_pixel_density[i]
	if (val < (v_mean - 2*v_stdd)) or (val > (v_mean + 2*v_stdd)):
		minwidth.append(i)

plt.subplot(2,1,1)
plt.plot(vertical_pixel_density);
plt.title('Vertical pixel density')

for i in range(0,len(seperators)):
	x = seperators[i]
	plt.plot([x,x],[0,height])





h_mean = np.mean(horizontal_pixel_density)
h_stdd = np.std(horizontal_pixel_density)

minheight = []
print "vertical mean: ", h_mean 
print "vertical stdv: ", h_stdd
for i in range(0,len(horizontal_pixel_density)):
	val = horizontal_pixel_density[i]
	if (val < (h_mean - 2*h_stdd)) or (val > (h_mean + 2*h_stdd)):
		minheight.append(i)


plt.subplot(2,1,2)
plt.plot(horizontal_pixel_density);
plt.title('Horizontal pixel density')
#plt.show();

print  "minwidth", minwidth, "minheight", minheight

for i in range(0,len(seperators)):
	for j in range(0,height-1):
		xval = seperators[i]
		#img[j][xval] = [0,0,0]

subimg = img[minheight[0]:height-1,0:width-1]


cv2.imshow('image',img)
cv2.waitKey(0)
#cv2.imshow('image',subimg)
#cv2.waitKey(0)
"""


"""
cv2.destroyAllWindows()
"""


if __name__ == "__main__":
	minBoxWidth = 75
	threshold = 4

	img = cv2.imread('example.pgm')
	img = rotateImage(img)

	h_pixel_density = calcHorPixelDensity(img)
	v_pixel_density = calcVerPixelDensity(img)
	seperators = CalcSeperators(v_pixel_density, minBoxWidth, threshold)

	drawDensities(h_pixel_density, v_pixel_density)

	drawSeperators(img, seperators)






