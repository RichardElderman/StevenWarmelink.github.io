from __future__ import print_function
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def readLabelledData():
	labelled_data = []
	for i, file in enumerate(os.listdir('Labelled')):
		print(i,  end="\r")
		if file.endswith('.pgm'):
			img = cv2.imread('Labelled/' + file, 0)
			utf = file[:4]
			labelled_data.append((utf,img))
	return labelled_data

def readData():
	train_data = []
	for i, file in enumerate(os.listdir('TestFolder')):
		print(i,  end="\r")
		if file.endswith('.jpg'):
			img = cv2.imread('TestFolder/' + file, 0)
			train_data.append(img)
	return train_data

def reLuMatrix(dat):
	# set all negative values in the matrix to zero
	return dat.clip(min=0)

def pooling(dat, window, stride):
	# slide window over dat in steps of stride, take in each window the maximum value
	nrow, ncol = np.shape(dat)
	prow = nrow/stride + (1 if nrow%stride>0 else 0)
	pcol = ncol/stride + (1 if ncol%stride>0 else 0)
	out = np.zeros(shape=(prow, pcol))
	for r in range(0,prow):
		for c in range(0,pcol):
			# make sure that at the end no indexes occur outside the shape of the matrix
			if(c==pcol-1):
				if(r==prow-1):
					out[r,c] = np.max(dat[r*stride:nrow, c*stride:ncol])
				else:
					out[r,c] = np.max(dat[r*stride:r*stride+window, c*stride:ncol])
			else:
				if(r==prow-1):
					out[r,c] = np.max(dat[r*stride:nrow, c*stride:c*stride+window])
				else:
					out[r,c] = np.max(dat[r*stride:r*stride+window, c*stride:c*stride+window])
	return out

def convolution(dat, mask):
	# nrow, ncol = np.shape(dat)
	# out = np.sum(np.multiply(dat, mask))
	# return out/(nrow*ncol)
	return np.mean(np.multiply(dat, mask))

def convLayer(dat, mask):
	nrow, ncol = np.shape(dat)
	rMask, cMask = np.shape(mask)
	out = np.zeros(shape=(nrow, ncol))
	for r in range(0,nrow):
		for c in range(0,ncol):
			sub = np.zeros(shape=(rMask, cMask)) 
			if(nrow-r<rMask):
				if(ncol-c<cMask):
					sub[0:nrow-r,0:ncol-c]=dat[r:nrow, c:ncol]
					out[r,c] = convolution(sub, mask)
				else:
					sub[0:nrow-r,0:cMask]=dat[r:nrow, c:c+cMask]
					out[r,c] = convolution(sub, mask)

			else:
				if(ncol-c<cMask):
					sub[0:rMask,0:ncol-c]=dat[r:r+rMask, c:ncol]
					out[r,c] = convolution(sub, mask)
				else:
					sub = dat[r:r+rMask, c:c+cMask]
					out[r,c] = convolution(sub, mask)
	return out

def neuralNetwork(nLayers, nNodes):
	return 0

def genInputLayer(img):
    layer = np.zeros(shape=(128, 128))
    width, hight = img.shape
    for i in range(1,width):
        for j in range(1,hight):
            layer[i][j]=img[i,j]

    return layer

def transformImage(input):
	ncol, nrow =input.shape
	out = np.zeros(shape=(nrow, ncol))
	for r in range(0, nrow):
		for c in range(0, ncol):
			if(input[r,c]==255):
				out[r,c] = -1
			else:
				out[r,c] = 1
	return out


# Xdata = np.matrix([[1,-1,-1,-1,1],
# 				   [-1,1,-1,1,-1],
# 				   [-1,-1,1,-1,-1],
# 				   [-1,1,-1,1,-1],
# 				   [1,-1,-1,-1,1]])

mask = np.matrix([[1,-1],
				  [-1,1]])

mask3 = np.matrix([[1,-1,-1],
				   [-1,1,-1],
				   [-1,-1,1]])

def predefinedFeature(dat):
	return 0



def performtest(dat):
    inputLayer = genInputLayer(dat)
    testim = transformImage(inputLayer)
    print(testim)
    convIm = convLayer(testim, mask3)
    print(convIm)
    print(np.unique(convIm))
    poolIm = pooling(convIm, 3, 3)
    print(poolIm)
    print(np.unique(poolIm))
    print(np.shape(poolIm))


    plt.subplot(2,1,1),plt.imshow(dat),plt.title("Original Image")
    plt.subplot(2,1,2),plt.imshow(testim),plt.title("Test")
    plt.show()
    plt.matshow(poolIm, fignum=10, cmap=plt.cm.gray),plt.title("Pool")
    plt.show()


	# Odata = np.matrix([[-1,-1,1,-1,-1],
# 				   [-1,1,-1,1,-1],
# 				   [1,-1,-1,-1,1],
# 				   [-1,1,-1,1,-1],
# 				   [-1,-1,1,-1,-1]])

# Rdata = np.matrix([[-3,5,6,2,1,8,9],
# 				   [-10,2,3,5,6,5,0],
# 				   [2,3,7,5,7,6,-1],
# 				   [14,12,12,12,6,5,-3]])

#print(Rdata)
#pX = pooling(Rdata, 2, 2)
#print(pX)
#prX = pooling(reLuMatrix(Rdata), 2, 2)
#print(prX)
# print("pattern X:")
# print(Xdata)
# print("mask:")
# print(mask)
# print("mask3:")
# print(mask3)
# Xconv = convLayer(Xdata, mask)
# print("Convolution 1:")
# print(Xconv)
# Xconv = convLayer(Xdata, mask3)
# print("Convolution 2:")
# print(Xconv)

# print("pattern O:")
# print(Odata)
# print("mask:")
# print(mask)
# print("mask3:")
# print(mask3)
# Xconv = convLayer(Odata, mask)
# print("Convolution 1:")
# print(Xconv)
# Xconv = convLayer(Odata, mask3)
# print("Convolution 2:")
# print(Xconv)
#Xconv3 = convLayer(Xdata, mask3)
#print(Xconv)
#print(Xconv3)
# pX3 = pooling(reLuMatrix(Xconv3), 2, 2)
# print(pX3)

if __name__ == "__main__":
    images = readData()



    for i in range(0, len(images)):
        performtest(images[i])
