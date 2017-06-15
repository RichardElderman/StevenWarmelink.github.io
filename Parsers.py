import cv2
import os
import numpy as np


#Reads all of the images and xml files
def readData(path):
	Images = {}
	for i, file in enumerate(os.listdir(path)):
		print(i,  end="\r")
		if file.endswith('.xml'):
			XML = open(path + '/'+ file, 'r').read()
			Images[XML] = []
			print(file)
			for j, file2 in enumerate(os.listdir(path)):
				if file2.endswith('.pgm') and file2[0] == file[0]:
					print('   ' + file2)
					img = cv2.imread(path + '/' + file, 0)
					Images[XML].append(img)
	return Images

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

#Parses the information from an xml file (String) to a dictionary
#with the useful information (name, x, y, w, h and utf)
def getLabelInfo(label):
	label_info = {}
	label_info['utf'] = None
	name = ''
	for a in label:
		if(a!= 'z'):
			name += a
		else:
			break
	label_info['name'] = name

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
		if(c == '<' and label[i+1]=='u' and label[i+2]=='t'and label[i+3]=='f' and label[i+4]=='>' and label[i+5] == ' '):
			value =''
			for a in label[i+6:]:
				if(a != ' '):
					value += a
				else:
					break
			label_info['utf'] = value
	return label_info

#Creates a new line for the XML file
def createLineXML(name, x, y, w, h, utf):
	new_line = name + 'zone-HUMAN-x=' + repr(x) + '-y=' +repr(y) + '-w=' + repr(w) +'-h=' +repr(h) + '-ybas=0000-nink=0000-segm=PERM1fwd <txt>@TAGGED_BY_TEAM_CRITICAL</txt> <utf> ' + utf + ' </uft>'
	return new_line 

#Replaces an line in the XML file with a new XML file
def replaceLineXML(XML, new_line, line_number):
	new_XML = ""
	for line_n, line in enumerate(XML.splitlines()):
		if line_n == 0:
			if line_n == line_number:
				new_XML =  new_line +'\n'
			else:
				new_XML =  line
		else:
			if line_n == line_number:
				new_XML =  new_XML + '\n' + new_line+'\n'
			else:
				new_XML =  new_XML + '\n' + line
	return new_XML

#Adds an XML line to the XML file
def addLineXML(XML, line):
	new_XML = XML + '\n' + line
	return new_XML

#Adds a UTF to the XML of a character at a particular position in the image
# (In terms of the first, second, third... character in the image)
def addUTF(XML, utf, position):
	old_line = XML.splitlines()[position]
	info = getLabelInfo(old_line)
	new_line = createLineXML(info['name'], info['x'], info['y'], info['w'], info['h'], utf)
	new_XML = replaceLineXML(XML, new_line, position)
	return new_XML

#Export the XML file
def exportXML(XML, name):
	print(XML)
	with open(name + ".xml", "w") as file:
	    file.write(XML)

if __name__ == "__main__":

	############## P1 ###################
	# This piece of code exports the XML
	# info and the images after segmentation.
	# This needs to be executed for every image
	##################################
	
	#Name can be retreived using getLabelInfo()
	name = 'something-' 
	#Number of the current image that is being segmented:
	img_num = 1
	#New XML file
	P1_XML = '\n'
	#For each segment in the image
	for segment in range(0,3):
		#Once these variables are found:
		x=y=w=h = 1
		line = createLineXML(name, x,y,w,h,'TBD')
		P1_XML = addLineXML(P1_XML, line)
	print(P1_XML)


	############## P2 ###################
	# This piece of code reads the input 
	# of the segmentation part and outputs
	# it into the input of the CNN.
	##################################

	#A list of XML, images pair. Each XML has a number of images
	XML_images = readData('segmentation')
	print(XML_images)


	############## P3 ###################
	# This piece of code exports the output
	# of the CNN to the desired XML format
	##################################

	for XML in XML_images:
		for position, images in enumerate(XML_images[XML]):
			#CNN classifies this image:
			utf = 'TBD'
			XML = addUTF(XML, utf, position)
		exportXML(XML, 'OUTPUT')


	########## TRAINING PARSER #########
	# This piece of code imports the labelled
	# data that is used for training. It is 
	# a list of (image, utf) tuples. 
	####################################
	labelled_data = readLabelledData()
