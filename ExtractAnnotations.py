import numpy as np
import cv2
import os

#Reads all of the images and xml files
def readData():
	paired_data = []
	img = cv2.imread('Train/navis-Ming-Qing_18341_0004-line-001-y1=0-y2=289.pgm')
	for file in os.listdir('Train'):
		print(file)
		if file.endswith('.pgm'):
			img = cv2.imread('Train/' + file)
		if file.endswith('.xml'):
			label = open('Train/'+ file, 'r').read()
			paired_data.append((img, label))
	return paired_data

#Parses the information from an xml file (String) to a dictionary
#with the useful information (x, y, w, h and utf)
def getLabelInfo(label):
	label_info = {}
	label_info['utf'] = None
	for i, c in enumerate(label):
		if(c == '-' and label[i+1]=='x' and label[i+2]=='='):
			value =''
			for a in label[i+3:]:
				if(a.isdigit()):
					value += a
				else:
					break
			label_info['x'] = value
		if(c == '-' and label[i+1]=='y' and label[i+2]=='='):
			value =''
			for a in label[i+3:]:
				if(a.isdigit()):
					value += a
				else:
					break
			label_info['y'] = value
		if(c == '-' and label[i+1]=='w' and label[i+2]=='='):
			value =''
			for a in label[i+3:]:
				if(a.isdigit()):
					value += a
				else:
					break
			label_info['w'] = value
		if(c == '-' and label[i+1]=='h' and label[i+2]=='='):
			value =''
			for a in label[i+3:]:
				if(a.isdigit()):
					value += a
				else:
					break
			label_info['h'] = value
		if(c == 'f' and label[i+1]=='>' and label[i+2]==' '):
			value =''
			for a in label[i+3:]:
				if(a != ' '):
					value += a
				else:
					break
			label_info['utf'] = value
	return label_info

#Cuts an image based on x, y, w and h values
def cutImage(image, label_info):
	x = int(label_info['x']);
	y = int(label_info['y']);
	w = int(label_info['w']);
	h = int(label_info['h']);
	cropped_image = image[y:y+h, x:x+w]
	#cv2.imshow("cropped", cropped_image)
	#cv2.waitKey(0)
	return cropped_image


#Extracts the annotated characters from the images and pairs them with
#their labels to create labelled_data
def extractAnnotatedSegments():
	paired_data = readData()
	labelled_data = []
	for pair in paired_data:
		image = pair[0]
		raw_xml = pair[1]
		xml_info = getLabelInfo(raw_xml)
		print(xml_info)
		if(xml_info['utf'] is None):
			continue
		else:
			label = xml_info['utf']
			cut_image = cutImage(image, xml_info)
			labelled_data.append([cut_image, label])
			cv2.imwrite('Labelled/'+label+'.pgm',cut_image)
extractAnnotatedSegments()

