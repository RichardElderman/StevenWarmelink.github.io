import cv2
import numpy as np
from matplotlib import pyplot as plt

def binarize(img):
	blur = cv2.GaussianBlur(img,(5,5),0)
	ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return otsu


def rotateImage(img):
	
	img = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_CONSTANT,value=[255,255,255])
	
	resultImage = img
	maxheight = 0

	for i in np.arange(-2,2,0.2):
		height, width = img.shape
		M = cv2.getRotationMatrix2D((width/2,height/2),i,1)
		rotatedImage = cv2.warpAffine(img,M,(width,height))

		height, width = img.shape	

		rotatedImage = rotatedImage[100:height-101,100:width-101]

		tempmax = max(calcHorPixelDensity(rotatedImage))

		if tempmax > maxheight:
			maxheight = tempmax
			resultImage = rotatedImage

	return np.asarray(resultImage)

def cropImage(img, h_thresh, v_thresh, h_dens, v_dens):
	height, width = img.shape

	vborders = []
	hborders = []

	subimg = img

	for i in range (0, width-1):
		if v_dens[i] > v_thresh:
			vborders.append(i)
	
	for j in range (0, height-1):
		if h_dens[j] > h_thresh:
			hborders.append(j)

	if len(hborders) > 0:
		if (max(hborders) - min(hborders)) > 100:
			subimg = subimg[min(hborders):max(hborders),0:width-1]
		else:
			if max(hborders) > height/1.5:
				subimg = subimg[0:min(hborders),0:width-1]	
			else: 
				subimg = subimg[max(hborders):height-1,0:width-1]

	if len(vborders) > 0:
		if (max(vborders) - min(vborders)) > 100:
			subimg = subimg[0:height-1,min(vborders):max(vborders)]
		else:
			if max(vborders) > width/2:
				subimg = subimg[0:height-1,0:min(vborders)]	
			else: 
				subimg = subimg[0:height-1,max(vborders):width-1]

	return np.asarray(subimg)



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

def CalcSeperators(vertical_pixel_density,minBoxWidth,threshold,padding):
	seperators = []

	prev_loc = 0

	for i in range (0,len(vertical_pixel_density)-1):
		if not ((i - prev_loc) < minBoxWidth):
			if vertical_pixel_density[i] < threshold:
				prev_loc = i
				if(i+padding) < len(vertical_pixel_density)-1:
					seperators.append(i+padding)

	return seperators

def drawSeperators(img, seperators, padding):
	height, width = img.shape

	for i in range(0,len(seperators)):
		for j in range(0,height-1):
			xval = seperators[i]
			if xval < width:
				img[j][xval] = 0

	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def splitImage(img, seperators):
	images = []

	height, width = img.shape
	
	if len(seperators) > 0:
		images.append(img[0:height-1,0:seperators[0]])
		
		for i in range (0,len(seperators)-2):
			images.append(img[0:height-1,seperators[i]:seperators[i+1]])

		images.append(img[0:height-1,seperators[len(seperators)-1]:width-1])
	return images

def showImages(images):
	for img in images:
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


if __name__ == "__main__":
	minBoxWidth = 75
	threshold = 10
	padding = 5

	horizontal_density_threshold = 1500
	vertical_density_threshold = 100 

	img = cv2.imread('example.pgm',0)
	img = binarize(img)
	img = rotateImage(img)


	h_pixel_density = calcHorPixelDensity(img)
	v_pixel_density = calcVerPixelDensity(img)
	
	img = cropImage(img,horizontal_density_threshold,vertical_density_threshold, h_pixel_density, v_pixel_density)
	
	h_pixel_density = calcHorPixelDensity(img)
	v_pixel_density = calcVerPixelDensity(img)
	
	seperators = CalcSeperators(v_pixel_density, minBoxWidth, threshold,padding)

	drawDensities(h_pixel_density, v_pixel_density)

	drawSeperators(img, seperators, padding)

	images = splitImage(img, seperators)

	showImages(images)