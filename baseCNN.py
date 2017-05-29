from __future__ import print_function
from random import shuffle
import cv2
import os
import numpy as np

#Reads all of the labelled images
def readLabelledData():
  maxFiles = 1000
  lbls = []
  for i, file in enumerate(os.listdir('labelled')):
    if file.endswith('.pgm'):
      utf = file[:4]
      if utf not in lbls:
        lbls.append(utf)
      if i==maxFiles:
        break;

  numlbl = len(lbls) # number of classes in the data set
  labelled_data = []
  for i, file in enumerate(os.listdir('labelled')):
    print(i,  end="\r")
    if file.endswith('.pgm'):
      img = cv2.imread('labelled/' + file, 0)
      utf = file[:4]
      output = getDesiredOutput(numlbl, lbls.index(utf))
      labelled_data.append((utf,img, output))
    if len(labelled_data)==maxFiles:
      break;
  return labelled_data, numlbl

def getDesiredOutput(length, onepos):
  out = np.zeros(shape=(1,length))
  out[0,onepos] = 1
  return out

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

def reLuMatrix(x):
  # use softplus method to be able to find derivative (needed for learning)
  return np.log(1+np.exp(x))
  # set all negative values in the matrix to zero
  # return x.clip(min=0)

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

# read imgs to classify, and get number of classes
labelled_data, totclasses = readLabelledData()
# shuffle images, such that classes are more evenly distributed across data set
shuffle(labelled_data)


imgWidth = 128
imgHeight = 128


np.random.seed(1)


# randomly initialized masks, currently not updated yet (hence bad classifications?)
mask = np.random.random((2, 2)) - 1
mask3 = np.random.random((3, 3)) - 1



# randomly initialize our weights with mean 0
# weights from input layer ("output of conv. layer" nodes) to hidden layer (4 nodes) fully connected
syn0 = 2 * np.random.random((1849, 8)) - 1
# weights from hidden layer (4 nodes) to output layer ("number of classes" nodes)
syn1 = 2 * np.random.random((8, totclasses)) - 1


for j in range(200):

    x = labelled_data[j]
    im = x[1]
    y = x[2]

    convIm = pooling(reLuMatrix(convLayer(im, mask3)), 3, 3)
    l0 = convIm.reshape(1,sum(len(x) for x in convIm))
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    l2_error = y - l2

    if (j % 1) == 0:
      print ("Error " + str(j) + " :" + str(np.mean(np.abs(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

totCorrect = 0
tot = 0
for j in range(200,300):
  x = labelled_data[j]
  tot = tot + 1
  im = x[1]
  y = x[2]
  convIm = pooling(reLuMatrix(convLayer(im, mask3)), 3, 3)
  l0 = convIm.reshape(1,sum(len(x) for x in convIm))
  l1 = nonlin(np.dot(l0, syn0))
  l2 = nonlin(np.dot(l1, syn1))
  if(np.argmax(l2)==np.argmax(y)):
    totCorrect = totCorrect + 1
  print("Percentage correct: "+str(float(totCorrect)/float(tot))+" ("+str(totCorrect)+"/"+str(tot)+")",  end="\r")
  
  # X = np.array([[[1,-1,-1,-1,1],
#                   [-1,1,-1,1,-1],
#                   [-1,-1,1,-1,-1],
#                   [-1,1,-1,1,-1],
#                    [1,-1,-1,-1,1]], 
#                  [[-1,-1,1,-1,-1],
#                   [-1,1,-1,1,-1],
#                    [1,-1,-1,-1,1],
#                   [-1,1,-1,1,-1],
#                   [-1,-1,1,-1,-1]], 
#                  [[-1,1,-1,-1,-1],
#                   [-1,1,-1,1,-1],
#                    [1,-1,-1,-1,1],
#                   [-1,1,-1,1,-1],
#                   [-1,-1,1,-1,-1]], 
#                  [[-1,-1,-1,-1,1],
#                   [-1,1,-1,1,-1],
#                    [1,-1,1,-1,-1],
#                   [-1,1,-1,1,-1],
#                    [1,-1,-1,-1,1]],
#                   [[1,-1,-1,-1,1],
#                   [-1,1,-1,1,-1],
#                   [-1,-1,-1,-1,-1],
#                   [-1,1,-1,1,-1],
#                    [1,-1,-1,-1,1]], 
#                  [[-1,-1,-1,-1,-1],
#                   [-1,1,-1,1,-1],
#                    [1,-1,-1,-1,1],
#                   [-1,1,-1,-1,-1],
#                   [-1,-1,1,-1,-1]],
#                   [[-1,-1,-1,-1,-1],
#                   [-1,1,-1,1,-1],
#                   [-1,-1,1,-1,-1],
#                   [-1,1,-1,1,-1],
#                    [1,-1,-1,-1,-1]], 
#                  [[-1,-1,1,-1,-1],
#                   [-1,1,-1,1,-1],
#                    [-1,-1,-1,-1,-1],
#                   [-1,1,-1,1,-1],
#                   [-1,-1,-1,-1,-1]]])

# y = np.array([[1,0],
#                [0,1],
#                [0,1],
#                [1,0],
#                [1,0],
#                [0,1],
#                [1,0],
#                [0,1]])
# X = np.array([[0, 0, 1],
#               [0, 1, 1],
#               [1, 0, 1],
#               [1, 1, 1]])

# mask = np.matrix([[1,-1],
#           [-1,1]])

# mask3 = np.matrix([[1,-1,-1],
#            [-1,1,-1],
#            [-1,-1,1]])