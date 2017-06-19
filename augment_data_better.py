import cv2
import os
import math
import csv
import pixelDensity
import histogram as hist
from random import shuffle
import numpy as np


#Reads all of the new unlabelled images
def readData():
	train_data = []
	num_files = len(os.listdir('TestFolder'))
	print('Reading train data:')
	for i, file in enumerate(os.listdir('TestFolder')):
		print(round((i/num_files)*100,2),  end="\r")
		if file.endswith('.jpg'):
			img = cv2.imread('TestFolder/' + file, 0)
			train_data.append(img)
	return train_data

#Reads all of the labelled images
def readLabelledData():
	labelled_data = []
	num_files = len(os.listdir('labelled'))
	print('Reading labelled data:')
	for i, file in enumerate(os.listdir('labelled')):
		print(round((i/num_files)*100,2),  end="\r")
		if file.endswith('.pgm'):
			img = cv2.imread('labelled/' + file, 0)
			utf = file[:4]
			labelled_data.append((utf,img))
	return labelled_data

#Reads all of the font images
def readFontData():
	labelled_data = []
	num_files = len(os.listdir('labelled/font_data'))
	print('Reading font data:')
	for i, file in enumerate(os.listdir('labelled/font_data')):
		print(round((i/num_files)*100,2),  end="\r")
		if file.endswith('.pgm'):
			img = cv2.imread('labelled/font_data/' + file, 0)
			utf = file[:4]
			labelled_data.append((utf,img))
	return labelled_data

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def rotate(img, degrees):
	o_rows,o_cols = img.shape
	img = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_CONSTANT,value=[255,255,255])
	rows,cols = img.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1)
	rot_img = cv2.warpAffine(img,M,(cols,rows))
	rot_img = rot_img[50:rows-51,50:cols-51]
	crop_img = pixelDensity.cropSquare(rot_img)
	#showImage(crop_img)
	return crop_img

#Augments images using rotation, erosion and dilation
def augmentImage(image, i, samples_generate):
	remainder = samples_generate%3
	rotation_indices = []
	erosion_indices = []
	dilation_indices = []

	num_rotations = ((samples_generate-remainder)/3)+remainder
	num_erosions = ((samples_generate-remainder)/3)
	num_dilations = ((samples_generate-remainder)/3)

	for x in range(1, int((samples_generate-remainder)/3)):
		erosion_indices.append(x)
	for x in range(int((samples_generate-remainder)/3)+1, int(2*(samples_generate-remainder)/3)):
		dilation_indices.append(x)
	for x in range(int(2*(samples_generate-remainder)/3)+1, samples_generate+remainder):
		rotation_indices.append(x)

	aug_image = image

	if i in erosion_indices:
		kernel_values = int(i*(10/num_erosions))
		kernel = np.ones((kernel_values,kernel_values),np.uint8)
		#print('EROSION: Kernel: ' + repr(kernel_values))
		aug_image = cv2.erode(image,kernel,iterations = 1)

	if i in dilation_indices:
		kernel_values = int((i-num_erosions)*(8/num_dilations))
		kernel = np.ones((kernel_values,kernel_values),np.uint8)
		#print('DILATION: Kernel: ' + repr(kernel_values))
		aug_image = cv2.dilate(image,kernel,iterations = 1)

	if i in rotation_indices:
		degrees = ((i-num_erosions-num_dilations)*(60/num_rotations))-30
		if degrees == 0:
			degrees = 5
		#print('ROTATION: Degrees: ' + repr(degrees))
		aug_image = rotate(image, degrees)

	#showImage(aug_image)

	return aug_image

if __name__ == "__main__":
	#The number of images per class:
	num_images = 50

	#The training data (with labels)
	labelled_data = readLabelledData()
	#The extra labelled data from the fonts
	#font_data = readFontData()						#########

	#You can add the font data to the train data:
	train_data = labelled_data                         ########+ font_data
	#train_data = labelled_data
	print(hist.histogramInformation(hist.generateHistogram(train_data)))

	#First undersample the data
	print('STARTING UNDERSAMPLING...')

	reduced_data = []
	utf_histogram = {}

	for utf,image in train_data:
		if utf in utf_histogram:
			if utf_histogram[utf] < num_images:
				utf_histogram[utf] = utf_histogram[utf] + 1
				reduced_data.append([utf, image])
		else:
			utf_histogram[utf] = 1
			reduced_data.append([utf, image])

	print('FINISHED UNDERSAMPLING')

	#hist.plotHistogram(utf_histogram)
	print(hist.histogramInformation(utf_histogram))

	#Oversample the data
	print('STARTING OVERSAMPLING...')
	
	oversampled_data = reduced_data
	new_utf_histogram = {}

	#Copy utf_histogram
	for utf in utf_histogram:
		new_utf_histogram[utf] = 0	#############utf_histogram[utf]

	#Variables used to indicate the progress
	total_data_size = len(utf_histogram)*num_images
	progress = 0

	first_seen_utf = {}
	for utf, image in reduced_data:
		progress +=1

		#If there are enough samples already, skip
		if new_utf_histogram[utf] == num_images:
			continue

		# write image itself too
		oversampled_data.append([utf, image])
		print("Writing: ", utf+'_'+str(new_utf_histogram[utf]) +'.pgm')
		cv2.imwrite('Aug_Train/' +utf+'_'+str(new_utf_histogram[utf]) +'.pgm',image)
		new_utf_histogram[utf] = new_utf_histogram[utf] + 1

		#Use this to divide the samples equally
		first = False
		if not(utf in first_seen_utf):
			first = True
			first_seen_utf[utf] = 'Seen'
		
		#Calculate how many extra samples are needed 
		samples_needed = num_images - utf_histogram[utf]
		print("samples_needed: ", samples_needed)
		#Calculate how many samples this image will generate:
		remainder = samples_needed%utf_histogram[utf]
		samples_generate = int((samples_needed-remainder)/utf_histogram[utf])
		print("samples_generate:", samples_generate)
		if(first):
			samples_generate = int(samples_generate + remainder)
		#print('This class (' + utf + ') needs ' + repr(samples_needed) + ' samples in total, so I this image will generate ' + repr(samples_generate) + ' samples because there are ' + repr(utf_histogram[utf]) + ' images')
		start = new_utf_histogram[utf]
		#Generate the necessary amount of samples
		for i in range(start, start+samples_generate):
			# print("ss", samples_generate)
			progress +=1
			new_image = augmentImage(image, i, samples_generate)
			oversampled_data.append([utf, new_image])
			print("Writing: ", utf+'_'+str(i) +'.pgm')
			cv2.imwrite('Aug_Train/' +utf+'_'+str(i) +'.pgm',new_image)

			print(repr(round(100*progress/total_data_size,2)), end="\r")

		new_utf_histogram[utf] = new_utf_histogram[utf] + samples_generate

	print('FINISHED OVERSAMPLING')
		
	generated_hist = hist.generateHistogram(oversampled_data)

	hist.plotHistogram(generated_hist)
	print('\n'+hist.histogramInformation(generated_hist))

	#hist.plotHistogram(new_utf_histogram)
	#print('\n'+hist.histogramInformation(new_utf_histogram))
