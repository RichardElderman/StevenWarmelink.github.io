import cv2
import os
import math
import pixelDensity
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

#Calculate the Hamming distance between two images
def calcHammingDistance(new_image, labelled_image):
	if(new_image.shape != labelled_image.shape):
		print('ERROR: Image dimensions must agree')
	hamming_distance = np.count_nonzero(new_image!=labelled_image)
	return hamming_distance

#Find the most commonly occuring classes in the top k matches
def mostCommonClass(top_k_matches):
	final_class = top_k_matches[0][0]
	class_histogram = {}
	for (utf, hamming_distance) in top_k_matches:
		if utf in class_histogram:
			class_histogram[utf] += 1
		else:
			class_histogram[utf] = 1
	max_class = max(class_histogram, key=class_histogram.get)
	#If all have a frequency of 1, take the one with the lowest hamming distance
	if(class_histogram[max_class] > 1):
		final_class = max_class
	return final_class



def xfoldKNN(x, k, labelled_data):
	print('Starting '+repr(x) +'-fold cross validation on KNN')
	total_acc = 0
	#The size of each fold
	fold_size = math.floor(len(labelled_data)/x)

	#x fold cross validation
	for x in range(0, x):
		test_data = labelled_data[x*fold_size:x*fold_size+fold_size]
		train_data = labelled_data[0:x*fold_size] + labelled_data[x*fold_size+fold_size:]
		accuracy = KNN(k, train_data, test_data)
		print('in round ' + repr(x) + ' ' + repr(round(accuracy,2)) + '% was correct')
		total_acc += accuracy
	print('in total ' + repr(round((total_acc/x),2)) + ' % was correct')



#The K Nearest Neightbour algorithm
def KNN(k, labelled_data, test_data):
	print('running KNN with K=' + repr(k))
	#A list to store the new images with a label
	final_list = []
	#Number of correct 
	correct = 0
	
	#For each new image in the test set
	for i, (new_utf,new_image) in enumerate(test_data):
		print((i/len(test_data)*100),  end="\r")
		distance_list = []
		#Calculate the distance between the new image and all of the labelled images
		for (labelled_utf, labelled_image) in labelled_data:
			distance_list.append((labelled_utf, calcHammingDistance(new_image, labelled_image)))
		#Sort the list of distances (lowest distance first)
		distance_list = sorted(distance_list,key=lambda x: x[1], reverse=False)
		top_k_matches = distance_list[:k]
		#Find the most common class in the top k labels from the distance list
		final_class = mostCommonClass(top_k_matches)
		if(new_utf == final_class):
			correct = correct + 1

		#Find a labelled image with the same utf code as the 'final class'
		#Store both the new image and the labelled image in the final list
		for (labelled_utf, labelled_image) in train_data:
			if(labelled_utf == final_class):
				final_list.append((new_image, labelled_image))
				break

	return (correct/len(test_data)*100)

if __name__ == "__main__":
	#K value for KNN
	k = 3
	#The training data (with labels)
	labelled_data = readLabelledData()
	#The extra labelled data from the fonts
	#font_data = readFontData()
	#The new test data
	test_data = readData()

	#You can add the font data to the train data:
	#train_data = labelled_data + font_data
	train_data = labelled_data

	total_correct = 0
	shuffle(train_data)
	#KNN(k, train_data, test_data)
	xfoldKNN(10, k, train_data)