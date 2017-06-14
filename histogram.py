import cv2
import os
import math
import pixelDensity
from random import shuffle
import pylab as pl
import numpy as np
import csv
import statistics

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

#Generate a histogram of all of the characters
def generateHistogram(total_data):
	histogram = {}
	for i, (utf,image) in enumerate(total_data):
		if utf in histogram:
			histogram[utf] = histogram[utf] + 1
		else:
			histogram[utf] = 1
	return histogram

#Normalize the histogram between 0 and 1
def normalizedHistogram(histogram):
	normalizedHistogram = {}
	max_value = histogram[max(histogram, key=histogram.get)]
	for key in histogram:
		normalizedHistogram[key] = histogram[key]/max_value
		#print(key + ' ' + repr(normalizedHistogram[key]))
	return normalizedHistogram

#Export the histogram to a csv
def exportHistogram(histogram):
	with open('histogram.csv', 'w', newline='') as csvfile:
		wr = csv.writer(csvfile)
		for key in histogram:
			wr.writerow([key, histogram[key]])

#Returns statistical information about the histogram
def histogramInformation(histogram):
	numbers = [histogram[key] for key in histogram]
	h_mean = statistics.mean(numbers)
	h_median = statistics.median(numbers)
	h_min = min(numbers)
	h_max = max(numbers)
	info = 'Histogram info:\n' + 'Labelled images: '+ repr(sum(numbers)) + '\nClasses: ' + repr(len(histogram)) +'\nMean: ' + repr(h_mean) + '\nMedian: ' + repr(h_median)+'\nMinimum: ' + repr(h_min) + '\nMaximum: ' + repr(h_max)

	return info

def plotHistogram(histogram):
	X = np.arange(len(histogram))
	pl.bar(X, sorted(histogram.values()), align='center', width=0.5)
	pl.xticks(X, histogram.keys())
	ymax = max(histogram.values()) + 1
	pl.ylim(0, ymax)
	pl.show()


if __name__ == "__main__":
	labelled_data = readLabelledData()

	total_data = labelled_data
	histogram = generateHistogram(total_data)
	#print(histogramInformation(histogram))
	plotHistogram(histogram)

	font_data = readFontData()
	total_data = labelled_data + font_data
	histogram = generateHistogram(total_data)
	plotHistogram(histogram)


	norm_histogram = normalizedHistogram(histogram)
	#print(repr(len(total_data)) + ' labelled images')
	#print(repr(len(histogram)) + ' classes')
	exportHistogram(histogram)