import cv2
import numpy as np
import math
import os
import sys

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
	angle = 0 
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
	return img, angle

def cropImage(img, h_thresh, v_thresh, h_dens, v_dens):
	height, width = img.shape

	padding = 10

	#print "Image height: ", height
	#print "Image width:  ", width
	#print "Max Vdens:    ", max(v_dens)
	#print "Max Hdens:    ", max(h_dens)
	#print "Vthresh:      ", v_thresh
	#print "Hthresh:      ", h_thresh


	h_lines = []
	v_lines = []
	# Detect whether there are horizontal lines, and where they are
	for i in range(0,height-1):
		if h_dens[i] > h_thresh:
			h_lines.append(i)

	#print "hlines: ", h_lines

	for i in range(0,width-1):
		if v_dens[i] > v_thresh:
			v_lines.append(i)

	#print "Vlines: ", v_lines

	# for i in range(h_lines[0]-10,h_lines[len(h_lines)-1]):
	# 	for j in range(1,width):
	# 		img[i,j] = 255

	# if (len(v_lines) > 0):
	# 	for i in range(v_lines[0]-10,v_lines[len(v_lines)-1]):
	# 		for j in range(1,height):
	# 			img[j,i] = 255			


	[x0, x1] = seqList(h_lines)
	[y0, y1] = seqList(v_lines)

	# print h_lines
	#showImages([img])

	subimg = img[x0-padding:x1+padding,0:width-1]

	# cv2.imshow("img",subimg)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	#showImages([subimg])

	#h_pixel_density = calcHorPixelDensity(subimg)
	#v_pixel_density = calcVerPixelDensity(subimg)
	#drawDensities(h_pixel_density,v_pixel_density)

	return subimg, [x0,x1]




def seqList(vh_list):
	max_index = 0
	max_len = 0
	newSequence = True

	temp_index = 0
	tempmax = 0

	for i in range (1,len(vh_list)):
		if (vh_list[i] == vh_list[i-1] + 1):
			if newSequence:
				temp_index = i
				newSequence = False 
		else: 
			if newSequence == False:
				newSequence = True
				tempmax = i - temp_index
				if tempmax > max_len:
					max_len = tempmax
					max_index = temp_index
		if i == len(vh_list)-1: 
			tempmax = i - temp_index
			if tempmax > max_len:
				max_len = tempmax
				max_index = temp_index

	# print "max_index:", max_index
	# print "max_len:  ", max_len
	# print "x1: ", vh_list[max_index-1]
	# print "x2: ", vh_list[(max_index-1)+(max_len)]

	return [vh_list[max_index-1], vh_list[(max_index-1)+(max_len)]]

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
		sum = 0
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

	return resultImages, rmv_array

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


def checkForMultipleCharacters(images):
	resultImages = []
	for img in images:
		height, width = img.shape
		if height > 1.5 * width:
			img1 = img[0:int((width/2)),0:height-1]
			img2 = img[int(width/2):int(width-1),9:height-1]
			resultImages.append(img1)
			resultImages.append(img2)
		else: 
			resultImages.append(img)

	return resultImages



# Function which splits the image in parts based on seperators, removes images with too many or few pixels 
# and resizes them to 128 x 128 pixels
def splitImage(img, seperators):
	images = []

	height, width = img.shape

	seperator_pairs = []
	
	if len(seperators) > 0:
		images.append(createSubImage(img,img[0:height-1,0:seperators[0]]))
		seperator_pairs.append([0, seperators[0]])	

		for i in range (0,len(seperators)-2):
			images.append(createSubImage(img,img[0:height-1,seperators[i]:seperators[i+1]]))	
			seperator_pairs.append([seperators[i], seperators[i+1]])	

		images.append(createSubImage(img,img[0:height-1,seperators[len(seperators)-2]:seperators[len(seperators)-1]]))	
		seperator_pairs.append([seperators[len(seperators)-2], seperators[len(seperators)-1]])

	images = checkForMultipleCharacters(images)

	images, rmv_array = removeSubimagesOutsideRange(images)
	


	res_seperator_pairs = []
	for i in range(0,len(seperator_pairs)):
		if not (i in rmv_array):
			res_seperator_pairs.append(seperator_pairs[i])
	
	#print(repr(res_seperator_pairs), repr(rmv_array))

	images = resizeImages(images)


	return images, res_seperator_pairs


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
		print("IMAGE NONEXISTENT (dimensions 0x0)")
		return None
	else:
		cropped_image = cv2.resize(cropped_image, (128, 128))
	
	return cropped_image

def warpToCoordinates(img_height, img_y, seperator_pairs):
	RoIs = []
	coordinateList = []

	for pair in seperator_pairs:
		RoIs.append([pair[0],img_y,pair[1]-pair[0],img_height])

	for sublist in RoIs:
		x=sublist[0]
		y=sublist[1]
		width=sublist[2]
		height=sublist[3]
		coordinateList.append([(x,y),(x,y+height),(x+width,y+height),(x+width,y)])

	return coordinateList

def getmin(list):
	min = 1000000
	for element in list:
		if element < min:
			min = element
	return min

def getmax(list):
	max = 0
	for element in list:
		if element > max:
			max = element
	return max


def rotateCoordinates(coordinateList,img_angle, img_center,img_height,img_width):
	# print(img_angle,img_center)
	middle_x = img_center[0]
	middle_y = img_center[1]

	rotatedList = []

	for box in coordinateList:
		tupleList = []
		for coordinate in box:
			x_old = coordinate[0]
			y_old = coordinate[1]
			delta_x = middle_x - x_old
			delta_y = middle_y - y_old 
			length = np.sqrt(delta_x*delta_x + delta_y*delta_y)
			alpha = np.rad2deg(np.arctan(delta_x / delta_y))

			beta = alpha - img_angle

			new_delta_x = -np.sin(np.deg2rad(beta)) * length
			new_delta_y = -np.cos(np.deg2rad(beta)) * length

			x_new = int(min(max(0,middle_x - new_delta_x),img_width))
			y_new = int(min(max(0,middle_y - new_delta_y),img_height))

			tupleList.append((x_new,y_new))

		rotatedList.append(tupleList)


	resultList = []
	for sublist in rotatedList:
		min_x = getmin([sublist[0][0],sublist[1][0],sublist[2][0],sublist[3][0]])
		max_x = getmax([sublist[0][0],sublist[1][0],sublist[2][0],sublist[3][0]])
		
		min_y = getmin([sublist[0][1],sublist[1][1],sublist[2][1],sublist[3][1]])
		max_y = getmax([sublist[0][1],sublist[1][1],sublist[2][1],sublist[3][1]])

		width = max_x - min_x
		height= max_y - min_y
		x = min_x
		y = min_y

		resultList.append((x,y,width,height))

	return resultList

#Creates a new line for the XML file
def createLineXML(name, x, y, w, h, utf):
	new_line = name + '-zone-HUMAN-x=' + repr(x) + '-y=' +repr(y) + '-w=' + repr(w) +'-h=' +repr(h) + '-ybas=0000-nink=0000-segm=PERM1fwd <txt>@TAGGED_BY_TEAM_CRITICAL</txt> <utf> ' + utf + ' </utf>'
	return new_line 


#Adds an XML line to the XML file
def addLineXML(XML, line):
	new_XML = XML + line + '\n' 
	return new_XML

#Export the XML file
def exportXML(XML, name):
	with open(name + ".xml", "w") as file:
	    file.write(XML)


def createXMLData(name, locationData):
	name = name 
	P1_XML = []
	#For each segment in the image
	for segment in locationData:
		#Once these variables are found:
		x = segment[0]
		y = segment[1]
		w = segment[2]
		h = segment[3] 
		P1_XML.append([name, x,y,w,h])
	return P1_XML

def createXMLFile(locationData, utfs):
	XML = ''
	i = 0
	name_out = ''
	#For each segment in the image
	for segment in locationData:
		#Once these variables are found:
		name = segment[0]
		name_out = name
		x = segment[1]
		y = segment[2]
		w = segment[3]
		h = segment[4] 
		utf = utfs[i]
		line = createLineXML(name, x,y,w,h,utf)
		XML = addLineXML(XML, line)
		i = i + 1
	exportXML(XML, name_out)

def showRoIs(rotatedList, inputImg):
	height, width = inputImg.shape 
	for v_pixel in range(0,width):
		for h_pixel in range(0,height):
			for list in rotatedList:
				# print(repr(list))
				if (v_pixel == list[0] or v_pixel == list[0]+list[2]) and h_pixel > list[1] and h_pixel < list[1]+list[3]:
					inputImg[h_pixel,v_pixel] = 0
				if (h_pixel == list[1] or h_pixel == list[1]+list[3]) and v_pixel > list[0] and v_pixel < list[0]+list[2]:
					inputImg[h_pixel,v_pixel] = 0
				
	writeImages([inputImg], "0000")

# Function which saves all images in list as jpg file in current folder
def writeImages(images, readstr):
	for i in range(0,len(images)):
		cv2.imwrite("segmentation/" + readstr + "_" + str(i).zfill(2) + ".jpg",images[i])

def loopthroughimages(readStr): 

	# Read image
	inputImg = cv2.imread(readStr,0)
	# Binarize image
	img = binarize(inputImg)

	height, width = img.shape
	img_center = (int(width/2),int(height/2))
	minBoxWidth = 75
	threshold = 12
	padding = 5

	horizontal_density_threshold = width / 20
	vertical_density_threshold = height / 10

	img, img_angle = houghRotation(img,rho=1,theta=np.pi/180,threshold=height+1,minLineLength=height+1,maxLineGap=20)
	#print("image angle :",img_angle)

	img = binarize(img)

	h_pixel_density = calcHorPixelDensity(img)
	v_pixel_density = calcVerPixelDensity(img)

	img, cropRegion = cropImage(img,horizontal_density_threshold,vertical_density_threshold, h_pixel_density, v_pixel_density)

	img_height = cropRegion[1] - cropRegion[0]
	img_y 	   = cropRegion[0]

	# print("image height:",repr(img_height))
	# print("image y     :",repr(img_y))

	h_pixel_density = calcHorPixelDensity(img)
	v_pixel_density = calcVerPixelDensity(img)
	
	seperators = CalcSeperators(v_pixel_density, minBoxWidth, threshold,padding)

	images, seperator_pairs = splitImage(img, seperators)

	coordinateList = warpToCoordinates(img_height,img_y,seperator_pairs)

	rotatedList = rotateCoordinates(coordinateList,img_angle,img_center,height, width)

	#showRoIs(rotatedList, inputImg)

	square_images = []
	for image in images:
		tempImg = cropSquare(image)
		if tempImg is not None:
			square_images.append(tempImg)


	xml_data = createXMLData(readStr, rotatedList)
	return square_images, xml_data



	
