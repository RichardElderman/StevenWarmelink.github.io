import cv2
import numpy as np
from matplotlib import pyplot as plt

def binarize(img):
	blur = cv2.GaussianBlur(img,(5,5),0)
	ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return otsu


def rotateImage(img):
	img = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_CONSTANT,value=[255,255,255])
	
	height, width = img.shape

	M = cv2.getRotationMatrix2D((width/2,height/2),1,1)
	img = cv2.warpAffine(img,M,(width,height))

	height, width = img.shape

	img = img[100:height-101,100:width-101]

	return np.asarray(img)

def calcVerPixelDensity(img):
	vertical_pixel_density = []

	height, width = img.shape
	for j in range(1,width):

		sum = 0;
		for i in range (1,height):
			px = img[i,j]
			if px == 0:
				sum +=1
		vertical_pixel_density.append(sum);

	return vertical_pixel_density

def calcHorPixelDensity(img):
	horizontal_pixel_density = []

	height, width = img.shape
	for j in range(1,height):

		sum = 0;
		for i in range (1,width):
			px = img[j,i]
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
	height, width = img.shape

	for i in range(0,len(seperators)):
		for j in range(0,height-1):
			xval = seperators[i]
			img[j][xval] = 0

	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


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

h_mean = np.mean(horizontal_pixel_density)
h_stdd = np.std(horizontal_pixel_density)

minheight = []
print "vertical mean: ", h_mean 
print "vertical stdv: ", h_stdd
for i in range(0,len(horizontal_pixel_density)):
	val = horizontal_pixel_density[i]
	if (val < (h_mean - 2*h_stdd)) or (val > (h_mean + 2*h_stdd)):
		minheight.append(i)

subimg = img[minheight[0]:height-1,0:width-1]
"""


if __name__ == "__main__":
	minBoxWidth = 75
	threshold = 10

	img = cv2.imread('example.pgm',0)
	img = binarize(img)
	img = rotateImage(img)

	h_pixel_density = calcHorPixelDensity(img)
	v_pixel_density = calcVerPixelDensity(img)
	seperators = CalcSeperators(v_pixel_density, minBoxWidth, threshold)

	drawDensities(h_pixel_density, v_pixel_density)

	drawSeperators(img, seperators)






