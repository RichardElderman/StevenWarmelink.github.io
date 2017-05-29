import pixelDensity
import cv2
import os

#Reads all of the labelled images
def FontToLabel(fontname, fontnumber):
	num_files = len(os.listdir('Labelled/' + fontname))
	for i, file in enumerate(os.listdir('Labelled/' + fontname)):
		print(round((i/num_files)*100,2),  end="\r")
		if file.endswith('.jpg'):
			img = cv2.imread('Labelled/'+fontname+'/' + file, 0)
			converted_image = pixelDensity.cropSquare(pixelDensity.binarize(img))
			utf = file[4:8]
			cv2.imwrite('Labelled/font_data/' +utf+'_'+str(i) +'.pgm',converted_image)

for i in range(1, 7):
	print('currently working on font ' + repr(i))
	FontToLabel('Font'+str(i), i)