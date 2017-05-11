import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import math

def binarize(img):
	blur = cv2.GaussianBlur(img,(5,5),0)
	ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return otsu


def houghRotation(img, rho, theta, threshold, minLineLength, maxLineGap):
	
	"""	cv2.imshow("img",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
	img = cv2.bitwise_not(img)

	lines = cv2.HoughLinesP(image=img,rho=1,theta=np.pi/180, threshold=threshold, minLineLength=minLineLength,maxLineGap=maxLineGap)

	img = cv2.bitwise_not(img)
	if len(lines) > 0:

		x1 = float(lines[0][0][0])
		y1 = float(lines[0][0][1])
		x2 = float(lines[0][0][2])
		y2 = float(lines[0][0][3])

		angle = math.degrees(math.atan((y2-y1)/(x2-x1)))

		img = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_CONSTANT,value=[255,255,255])

		height, width = img.shape
		M = cv2.getRotationMatrix2D((width/2,height/2),angle,1)
		img = cv2.warpAffine(img,M,(width,height))

		height, width = img.shape	

		img = img[100:height-101,100:width-101]
	"""
	cv2.imshow("img",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
	return img


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
			subimg = subimg[min(hborders)+10:max(hborders)-10,0:width-1]
		else:
			if max(hborders) > height/1.5:
				subimg = subimg[0:min(hborders)-10,0:width-1]	
			else: 
				subimg = subimg[max(hborders)+10:height-1,0:width-1]

	if len(vborders) > 0:
		if (max(vborders) - min(vborders)) > 100:
			subimg = subimg[0:height-1,min(vborders)+10:max(vborders)-10]
		else:
			if max(vborders) > width/1.5:
				subimg = subimg[0:height-1,0:min(vborders)-10]	
			else: 
				subimg = subimg[0:height-1,max(vborders)+10:width-1]

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
				if (i + minBoxWidth) < len(vertical_pixel_density)-1:
					sum = 0
					for j in range (i,i+minBoxWidth):
						sum += vertical_pixel_density[j]
					if sum > 20:
						prev_loc = i
						if(i+padding) < len(vertical_pixel_density)-1:
							seperators.append(i+padding)

	return seperators

def drawSeperators(img, seperators, padding):
	tempImg = img
	height, width = tempImg.shape

	for i in range(0,len(seperators)):
		for j in range(0,height-1):
			xval = seperators[i]
			if xval < width:
				tempImg[j][xval] = 0

	cv2.imshow('image',tempImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def createSubImage(img,subimg):
	height, width = img.shape	
	subimg = cv2.copyMakeBorder(subimg,100,100,100,100,cv2.BORDER_CONSTANT,value=[255,255,255])
	
	height, width = subimg.shape
	M = cv2.getRotationMatrix2D((width/2,height/2),270,1)
	rotatedImage = cv2.warpAffine(subimg,M,(width,height))
	
	height, width = rotatedImage.shape
	rotatedImage = rotatedImage[100:height-101,80:width-81]

	return rotatedImage


def removeSubimagesOutsideRange(images):

	rmv_array = []

	for i in range (0,len(images)-1):
		total = sum(calcHorPixelDensity(images[i]))

		height, width = images[i].shape
		if total < 0.02*height*width or total > 0.60*height*width:
			rmv_array.append(i)

	rmv_array = sorted(rmv_array, reverse=True)

	resultImages = []

	for i in range(0,len(images)-1):
		if not (i in rmv_array):
			resultImages.append(images[i])

	return resultImages


def resizeImages(images):

	resizedImages = []

	for image in images:
		height, width = image.shape
		if height > 128:
			height, width = image.shape
			h_diff = height - 128
			if h_diff%2 == 0:
				image = image[int(h_diff/2):height-int(h_diff/2),0:width]
			else:
				image = image[1+int(h_diff/2):height-int(h_diff/2),0:width]

		if height < 128:
			height, width = image.shape
			h_diff = 128 - height
			if h_diff%2 == 0:
				image = cv2.copyMakeBorder(image,int(h_diff/2),int(h_diff/2),0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
			else:
				image = cv2.copyMakeBorder(image,int(1+h_diff/2),int(h_diff/2),0,0,cv2.BORDER_CONSTANT,value=[255,255,255])

		if width > 128:
			height, width = image.shape
			w_diff = width - 128
			if w_diff%2 == 0:
				image = image[0:height,int(w_diff/2):width-int(w_diff/2)]
			else: 
				image = image[0:height,1+int(w_diff/2):width-int(w_diff/2)]		

		if width < 128:
			height, width = image.shape
			w_diff = 128 - width
			if w_diff%2 == 0:
				image = cv2.copyMakeBorder(image,0,0,int(w_diff/2),int(w_diff/2),cv2.BORDER_CONSTANT,value=[255,255,255])
			else:
				image = cv2.copyMakeBorder(image,0,0,1+int(w_diff/2),int(w_diff/2),cv2.BORDER_CONSTANT,value=[255,255,255])
		resizedImages.append(image)
	return resizedImages

def splitImage(img, seperators):
	images = []

	height, width = img.shape
	
	if len(seperators) > 0:
		images.append(createSubImage(img,img[0:height-1,0:seperators[0]]))	

		for i in range (0,len(seperators)-2):
			images.append(createSubImage(img,img[0:height-1,seperators[i]:seperators[i+1]]))	

		images.append(createSubImage(img,img[0:height-1,seperators[len(seperators)-1]:width-1]))	

	images = removeSubimagesOutsideRange(images)

	images = resizeImages(images)

	return images

def showImages(images):
	for img in images:
		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def writeImages(images):
	for i in range(0,len(images)-1):
		cv2.imwrite("character_" + str(i) + ".jpg",images[i])

def drawDensities(h_pixel_density, v_pixel_density):
	plt.subplot(2,1,1)
	plt.plot(v_pixel_density);
	plt.title('Vertical pixel density')

	plt.subplot(2,1,2)
	plt.plot(h_pixel_density);
	plt.title('Horizontal pixel density')
	plt.show();

#Crop/enlarge the image such that it fits within a 128x128 square
def cropSquare(image):
	height, width = image.shape
	min_horizontal_pixel = width
	max_horizontal_pixel = 0
	min_vertical_pixel = height
	max_vertical_pixel = 0

	#Find the minimal and maximal black pixel values on both the vertical and horizontal axis. 
	for v in range (0,height-1):
		for h in range(0,width-1):
			px = image[v,h]
			if px == 0 and h < min_horizontal_pixel:
				min_horizontal_pixel = h
			if px == 0 and h > max_horizontal_pixel:
				max_horizontal_pixel = h
			if px == 0 and v < min_vertical_pixel:
				min_vertical_pixel = v
			if px == 0 and v > max_vertical_pixel:
				max_vertical_pixel = v

	#Crop the image using the previously found pixel values. This effectively removes the white borders around the characters.
	cropped_image = image[min_vertical_pixel:max_vertical_pixel, min_horizontal_pixel:max_horizontal_pixel]

	#Pad the image with white pixels on either the vertical or horizontal sides such that the image becomes a square
	if((max_vertical_pixel-min_vertical_pixel) > (max_horizontal_pixel-min_horizontal_pixel)):
		w_diff = (max_vertical_pixel-min_vertical_pixel)-(max_horizontal_pixel-min_horizontal_pixel)
		if w_diff%2 == 0:
			cropped_image = cv2.copyMakeBorder(cropped_image,0,0,int(w_diff/2),int(w_diff/2),cv2.BORDER_CONSTANT,value=[255,255,255])
		else:
			cropped_image = cv2.copyMakeBorder(cropped_image,0,0,1+int(w_diff/2),int(w_diff/2),cv2.BORDER_CONSTANT,value=[255,255,255])
	if((max_horizontal_pixel - min_horizontal_pixel) > (max_vertical_pixel - min_vertical_pixel)):
		h_diff = (max_horizontal_pixel - min_horizontal_pixel) - (max_vertical_pixel - min_vertical_pixel)
		if h_diff%2 == 0:
			cropped_image = cv2.copyMakeBorder(cropped_image,int(h_diff/2),int(h_diff/2),0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
		else:
			cropped_image = cv2.copyMakeBorder(cropped_image,int(1+h_diff/2),int(h_diff/2),0,0,cv2.BORDER_CONSTANT,value=[255,255,255])

	#Resize the image to 128x128 pixels
	cropped_image = cv2.resize(cropped_image, (128, 128))
	return cropped_image


if __name__ == "__main__":
	minBoxWidth = 75
	threshold = 10
	padding = 5

	horizontal_density_threshold = 1500
	vertical_density_threshold = 100 

	img = cv2.imread('example.pgm',0)
	img = binarize(img)
	
	img = houghRotation(img,rho=1,theta=np.pi/180,threshold=400,minLineLength=500,maxLineGap=20)
	#img = rotateImage(img)


	h_pixel_density = calcHorPixelDensity(img)
	v_pixel_density = calcVerPixelDensity(img)

	#drawDensities(h_pixel_density, v_pixel_density)
	
	img = cropImage(img,horizontal_density_threshold,vertical_density_threshold, h_pixel_density, v_pixel_density)
	
	h_pixel_density = calcHorPixelDensity(img)
	v_pixel_density = calcVerPixelDensity(img)
	
	seperators = CalcSeperators(v_pixel_density, minBoxWidth, threshold,padding)

	#drawDensities(h_pixel_density, v_pixel_density)



	#drawSeperators(img, seperators, padding)

	images = splitImage(img, seperators)

	#Fit each image/character into a 128 by 128 square.
	square_images = []
	for image in images:
		square_images.append(cropSquare(image))
	

	"""plt.subplot(len(images),1,1),plt.imshow(img,'gray')
	for i in range(0,len(images)-1):	
		plt.subplot(len(images),2,i+2),plt.imshow(images[i],'gray')
		plt.xticks([]),plt.yticks([])
	plt.show()"""

	showImages(square_images)

	writeImages(square_images)