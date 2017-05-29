import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import math

# Function which binarizes the image by performing Otsu binarization after applying 
# a guassian blur. 
def binarize(img):
	# Performs gaussian blurring with a kernel size of (5,5)
	blur = cv2.GaussianBlur(img,(5,5),0)
	# Performs Otsu thresholding (binarization) on the blurred image
	ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return otsu


# Function which rotates the image based on lines in the image. The function detects all lines 
# with the given minimum length and maximum gap in between line segments, calculates the angle
# and then rotates the image based on the first line in the list.
def houghRotation(img, rho, theta, threshold, minLineLength, maxLineGap):
	
	# Invert the image for hough line function
	img = cv2.bitwise_not(img)

	# Calculate all hough lines using opencv2. minLineLength should be greater than the height 
	# of the image in order to prevent the algorithm from detecting vertical lines and rotating 
	# along those lines rather than horizontal lines. 

	# TODO:: Make houghRotation function dynamic based on image dimensions rather than using 
	# static minimum line length and maximum line gap.
	lines = cv2.HoughLinesP(image=img,rho=1,theta=np.pi/180, threshold=threshold, minLineLength=minLineLength,maxLineGap=maxLineGap)

	# Invert image again for original image
	img = cv2.bitwise_not(img)

	# If lines of sufficient length are detected, calculate the angle using the resulting 
	# coordinates. 
	if lines is not None:
		x1 = float(lines[0][0][0])
		y1 = float(lines[0][0][1])
		x2 = float(lines[0][0][2])
		y2 = float(lines[0][0][3])

		angle = math.degrees(math.atan((y2-y1)/(x2-x1)))

		# Before rotating, pad the image with 100 white pixels in all four directions to 
		# prevent black areas from showing up after rotation
		img = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_CONSTANT,value=[255,255,255])

		# Calculate the rotationmatrix based on the angle and image dimensions; Afterwards 
		# perform affine warp to rotate image
		height, width = img.shape
		M = cv2.getRotationMatrix2D((width/2,height/2),angle,1)
		img = cv2.warpAffine(img,M,(width,height))

		height, width = img.shape	

		# Crop out the 100 outer pixels again
		img = img[100:height-101,100:width-101]
	return img


# NOTE:: NO LONGER USED IN PIPELINE! MOSTLY REPLACED BY HOUGH TRANSFORM
# Function which rotates the image rotating in the range [-2,2] degrees in steps 
# of 0.2 degrees and returns the rotated image with the maximal horizontal density peak
# NOTE:: NO LONGER USED IN PIPELINE! MOSTLY REPLACED BY HOUGH TRANSFORM
def rotateImage(img):
	
	# Pad the image with 100 white pixels in all 4 directions to prevent black areas from showing up when rotating
	img = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_CONSTANT,value=[255,255,255])
	
	resultImage = img
	maxheight = 0

	# Rotate the image in the range [-2,2] degrees in steps of 0.2 degrees and calculate the highest horizontal 
	# pixel density. If this pixel density peak is higher than one of the previously calculated maximum densities, 
	# set the new highest peak to this value and set the result image to the current (rotated) image
	for i in np.arange(-2,2,0.2):
		height, width = img.shape
		M = cv2.getRotationMatrix2D((width/2,height/2),i,1)
		rotatedImage = cv2.warpAffine(img,M,(width,height))

		height, width = img.shape	

		# Before we save the image, we crop out the outer 100 pixels in all four directions again
		rotatedImage = rotatedImage[100:height-101,100:width-101]

		tempmax = max(calcHorPixelDensity(rotatedImage))

		if tempmax > maxheight:
			maxheight = tempmax
			resultImage = rotatedImage

	return np.asarray(resultImage)


# Function which attempts to detect horizontal and vertical lines and crops out the area 
# which contains the characters but not the horizontal/vertical lines. 
# TODO:: Improve function to take density curves into account when determining 
# 		 which area is most probable to contain the actual characters
# TODO:: Make thresholds dependent on image rather than static values
def cropImage(img, h_thresh, v_thresh, h_dens, v_dens):
	height, width = img.shape

	vborders = []
	hborders = []

	subimg = img

	# Add all locations where the vertical density is above the threshold to a list
	for i in range (0, width-1):
		if v_dens[i] > v_thresh:
			vborders.append(i)
	
	# Add all locations where the horizontal density is above the threshold to a list
	for j in range (0, height-1):
		if h_dens[j] > h_thresh:
			hborders.append(j)

	# If horizontal/vertical lines have been detected, check whether there are lines which are 
	# more than 100 pixels apart. If there are, we have more than one line 
	# segment, If there are no lines more than 100 pixels apart we (probably wrongly)
	# assume there is just one line segment. In case of more than one line segment
	# we take the area in between both lines. Otherwise we guess where characters are 
	# more likely to be and split either take a subimage of the area to the left or 
	# the right of the line segment. 

	# TODO:: Improve this part to take density curves into account when trying to determine 
	# 		 what part of the image is most likely to contain the characters. 
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




# Function which, given an image, returns a list of vertical pixel densities.
def calcVerPixelDensity(img):
	vertical_pixel_density = []
	height, width = img.shape

	# Walk across the image and calculate pixel densities by summing up black pixels
	for j in range(1,width):
		sum = 0;
		for i in range (1,height):
			px = img[i,j]
			if px == 0:
				sum +=1
		vertical_pixel_density.append(sum);

	return vertical_pixel_density


# Function which, given an image, returns a list of horizontal pixel densities.
def calcHorPixelDensity(img):
	horizontal_pixel_density = []
	height, width = img.shape

	# Walk across the image and calculate pixel densities by summing up black pixels
	for j in range(1,height):
		sum = 0;
		for i in range (1,width):
			px = img[j,i]
			if px == 0:
				sum +=1
		horizontal_pixel_density.append(sum);

	return horizontal_pixel_density


# Function which, given a list of vertical pixel densities, calculates seperator 
# locations based on a minimum box width (characters can not follow eachother within 
# N pixels), threshold (maximum pixel density allowed for a seperator to be allowed) 
# and padding (space between detected seperator location and returned seperator location
# - used to prevent cutting off parts of low-density characters)
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
					# Only add a seperator if the total number of pixels in the area
					# is greater than 20; should be made dynamic. 
					if sum > 20:
						prev_loc = i
						if(i+padding) < len(vertical_pixel_density)-1:
							seperators.append(i+padding)
	seperators.append(len(vertical_pixel_density)-1)

	return seperators

# Function which visualizes the seperators in the rotated, cropped image
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

# Function which returns a 90 degree rotated version of the subimg region of the input image
def createSubImage(img,subimg):
	height, width = img.shape	
	subimg = cv2.copyMakeBorder(subimg,100,100,100,100,cv2.BORDER_CONSTANT,value=[255,255,255])
	
	height, width = subimg.shape
	M = cv2.getRotationMatrix2D((width/2,height/2),270,1)
	rotatedImage = cv2.warpAffine(subimg,M,(width,height))
	
	height, width = rotatedImage.shape
	rotatedImage = rotatedImage[100:height-101,80:width-81]

	return rotatedImage

# Function which removes images with too many or too few pixels (less than 2% black pixels or more 
# than 60% black pixels in the entire image)
def removeSubimagesOutsideRange(images):
	rmv_array = []

	for i in range (0,len(images)-1):
		total = sum(calcHorPixelDensity(images[i]))

		height, width = images[i].shape
		if total < 0.02*height*width or total > 0.60*height*width:
			rmv_array.append(i)

	rmv_array = sorted(rmv_array, reverse=True)
	resultImages = []

	for i in range(0,len(images)):
		if not (i in rmv_array):
			resultImages.append(images[i])

	return resultImages

# Resizes all images to 128 x 128 pixel size. 
def resizeImages(images):
	resizedImages = []

	for image in images:
		height, width = image.shape
		
		# if image height is too large, take a subimage which is evenly cropped from top and bottom
		if height > 128:
			height, width = image.shape
			h_diff = height - 128
			if h_diff%2 == 0:
				image = image[int(h_diff/2):height-int(h_diff/2),0:width]
			else:
				image = image[1+int(h_diff/2):height-int(h_diff/2),0:width]

		# if image height is too small, evenly pad the top and bottom parts of image with white pixels
		if height < 128:
			height, width = image.shape
			h_diff = 128 - height
			if h_diff%2 == 0:
				image = cv2.copyMakeBorder(image,int(h_diff/2),int(h_diff/2),0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
			else:
				image = cv2.copyMakeBorder(image,1+int(h_diff/2),int(h_diff/2),0,0,cv2.BORDER_CONSTANT,value=[255,255,255])

		# if image width is too large, take a subimage which is evenly cropped from left and right
		if width > 128:
			height, width = image.shape
			w_diff = width - 128
			if w_diff%2 == 0:
				image = image[0:height,int(w_diff/2):width-int(w_diff/2)]
			else: 
				image = image[0:height,1+int(w_diff/2):width-int(w_diff/2)]		

		# if image width is too small, evenly pad the left and right parts of image with white pixels
		if width < 128:
			height, width = image.shape
			w_diff = 128 - width
			if w_diff%2 == 0:
				image = cv2.copyMakeBorder(image,0,0,int(w_diff/2),int(w_diff/2),cv2.BORDER_CONSTANT,value=[255,255,255])
			else:
				image = cv2.copyMakeBorder(image,0,0,1+int(w_diff/2),int(w_diff/2),cv2.BORDER_CONSTANT,value=[255,255,255])

		# add resized image to list of resized images
		resizedImages.append(image)

	return resizedImages


# Function which splits the image in parts based on seperators, removes images with too many or few pixels 
# and resizes them to 128 x 128 pixels
def splitImage(img, seperators):
	images = []

	height, width = img.shape
	
	if len(seperators) > 0:
		images.append(createSubImage(img,img[0:height-1,0:seperators[0]]))	

		for i in range (0,len(seperators)-2):
			images.append(createSubImage(img,img[0:height-1,seperators[i]:seperators[i+1]]))	

		images.append(createSubImage(img,img[0:height-1,seperators[len(seperators)-2]:seperators[len(seperators)-1]]))	

	images = removeSubimagesOutsideRange(images)
	
	images = resizeImages(images)

	return images

# Function which shows all images in list
def showImages(images):
	for img in images:
		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

# Function which saves all images in list as jpg file in current folder
def writeImages(images, imgcount):
	for i in range(0,len(images)-1):
		cv2.imwrite("images/character_" + str(imgcount) + '_' + str(i) + ".jpg",images[i])

# Function which visualizes horizontal and vertical pixel densities
def drawDensities(h_pixel_density, v_pixel_density):
	plt.subplot(2,1,1)
	plt.plot(v_pixel_density);
	plt.title('Vertical pixel density')

	plt.subplot(2,1,2)
	plt.plot(h_pixel_density);
	plt.title('Horizontal pixel density')
	plt.show();


# Function which crops/enlarges the image such that it fits within a 128x128 square
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
	height, width = cropped_image.shape
	if(height == 0 or width == 0):
		print "IMAGE NONEXISTANT (dimensions 0x0)"
	else:
		cropped_image = cv2.resize(cropped_image, (128, 128))
	
	return cropped_image



if __name__ == "__main__":
	
	for imagecount in range (1,10):

		readStr = 'example' + str(imagecount) + '.pgm' 

		img = cv2.imread(readStr,0)
		img = binarize(img)
		height, width = img.shape

		# Minimum distance between two seperators
		minBoxWidth = 75
		# Pixeldensity has to be below this threshold to trigger a seperator
		threshold = 10
		# Pixels padded after seperator has been detected (to prevent cutoffs)
		padding = 5
		# Pixel density threshold for cropping horizontal lines
		# TODO:: Replace by dynamic implementation  
		horizontal_density_threshold = width/2
		# Pixel density threshold for cropping vertical lines
		# TODO:: Replace by dynamic implementation
		vertical_density_threshold = height/2 


		# In order, read, binarize, rotate, crop, seperate, split, show and write the image.

		img = houghRotation(img,rho=1,theta=np.pi/180,threshold=height+1,minLineLength=height+1,maxLineGap=20)

		h_pixel_density = calcHorPixelDensity(img)
		v_pixel_density = calcVerPixelDensity(img)

		#drawDensities(h_pixel_density, v_pixel_density)
		
		img = cropImage(img,horizontal_density_threshold,vertical_density_threshold, h_pixel_density, v_pixel_density)
		
		#cv2.imshow('image',img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		h_pixel_density = calcHorPixelDensity(img)
		v_pixel_density = calcVerPixelDensity(img)
		
		seperators = CalcSeperators(v_pixel_density, minBoxWidth, threshold,padding)

		#drawDensities(h_pixel_density, v_pixel_density)

		#drawSeperators(img, seperators, padding)

		images = splitImage(img, seperators)

		#showImages(images)


		square_images = []
		for image in images:
			square_images.append(cropSquare(image))

		#showImages(square_images)

		writeImages(square_images, imagecount)