from __future__ import print_function
from random import shuffle
import cv2
import os
import numpy as np

#Reads all of the labelled images, and determined all different classes
def readLabelledData(maxFiles):
  lbls = []
  for i, file in enumerate(sorted(os.listdir('Labelled'))):
    if file.endswith('.pgm'):
      utf = file[:4]
      if utf not in lbls:
        lbls.append(utf)
      if i==maxFiles:
        break;

  numlbl = len(lbls) # number of classes in the data set
  labelled_data = []
  for i, file in enumerate(sorted(os.listdir('Labelled'))):
    print(i,  end="\r")
    if file.endswith('.pgm'):
      img = cv2.imread('Labelled/' + file, 0)
      utf = file[:4]
      output = getDesiredOutput(numlbl, lbls.index(utf))
      labelled_data.append((utf,img, output))
    if len(labelled_data)==maxFiles:
      break;
  return labelled_data, lbls

# Generate target output for an image
# Args: length of target vector, index in vector that must have value 1
# Returns vector of zeros of given length, with 1 at given position
def getDesiredOutput(length, onepos):
  out = np.zeros(shape=(1,length))
  out[0,onepos] = 1
  return out

# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

# Function for the ReLu layer, implemented using the Softplus function
# Args: input matrix/vector, boolean for derivative
def reLuLayer(x, deriv = False):
  # set every negative value in x to zero
  # use softplus method to be able to find derivative, which is nonlin with deriv=true (needed for learning)
  if(deriv == True):
    return nonlin(x, True)

  return np.log(1+np.exp(x))


# Function used for backpropagation through a pooling layer
# invert pooling by reconstructing original matrix, but only keeping values selected as local maxima
# Reshape to vector, and construct another vector with the errors at the right places
# Args: original input matrix to pool layer, matrix with indications of max places, vector of errors
# Returns reconstructed matrix as vector, and another vector of same size with errors
def invertPool(original, maxplaces, errors):
  only_maxes = np.multiply(original,maxplaces) # replace all irrelevant values (nonmax) with zero
  only_maxes_valuevec = only_maxes.reshape(1,sum(len(x) for x in only_maxes)) # reshape to vector

  # construct vector of size only_maxes_vec, with errors at max places
  maxplaces_vec = maxplaces.reshape(1,sum(len(x) for x in maxplaces))
  only_maxes_errorvec = np.zeros(shape=(1,len(only_maxes_valuevec)))
  error_indx = 0
  for indx in range(len(maxplaces_vec)):
    if maxplaces_vec[0, indx]==1: # indx is a local maxima: next error in errors belongs to indx
      only_maxes_errorvec[0, indx] = errors[0, error_indx]
      error_indx +=1
  return only_maxes_valuevec, only_maxes_errorvec
  

# Function for the max pooling layer. Args = dat-matrix, window size (square), stride size
# slide window over dat in steps of stride, take in each window the maximum value
# return matrix with max values, and a matrix of same size as dat with indications of max places
def maxPoolLayer(dat, window, stride):
  nrow, ncol = np.shape(dat)
  prow = nrow/stride + (1 if nrow%stride>0 else 0)
  pcol = ncol/stride + (1 if ncol%stride>0 else 0)
  out = np.zeros(shape=(prow, pcol))
  places = np.zeros(shape=(nrow, ncol)) # dat matrix of zeros, with 1 at places of max values
  for r in range(0,prow):
    for c in range(0,pcol):
      # make sure that at the end no indexes occur outside the shape of the matrix
      sub = dat[r*stride:min(r*stride+window, nrow), c*stride:min(c*stride+window, ncol)]
      out[r,c] = np.max(sub)
      max_r, max_c = np.unravel_index(sub.argmax(), np.shape(sub))
      places[(r*stride)+max_r, (c*stride)+max_c] = 1 # put 1 at place of maximal value
  return out, places

# Performs the actual convolution of dat and mask (assumption = dat and mask have same shape)
def convolution(dat, mask):
  return np.mean(np.multiply(dat, mask))

# Main function of a convolution layer: slides mask over dat, at each place 
# doing a convolution with the part of dat under the mask.
def convLayer(dat, mask):
  nrow, ncol = np.shape(dat)
  rMask, cMask = np.shape(mask)
  out = np.zeros(shape=(nrow, ncol))
  for r in range(0,nrow):
    for c in range(0,ncol):
      sub = np.zeros(shape=(rMask, cMask)) 
      sub[0:min(nrow-r, rMask),0:min(ncol-c, cMask)] = dat[r:min(nrow, r+rMask), c:min(ncol, c+cMask)]
      out[r,c] = convolution(sub, mask)

  return out

# Height and width of an input image
imgWidth = imgHeight = 128
# initialize random generator
np.random.seed(1)


# Randomly initialized masks, currently not updated yet (hence bad classifications?):
mask = np.random.random((2, 2)) - 1
mask3 = np.random.random((3, 3)) - 1

# Masks with predetermined values:
mask3 = np.matrix([[1,-1,-1],
           [-1,1,-1],
           [-1,-1,1]])
mask10 = np.matrix([
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [1,1,1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1,1,1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
print(convLayer(mask10, mask3))
# read 1000 images from data set, and get list of all classes (utf codes)
labelled_data, allclasses = readLabelledData(1000)
print("Number of classes: "+str(len(allclasses)))
# shuffle images, such that classes are more evenly distributed across data set
shuffle(labelled_data)

# list of parameters
inputlayer = 1849  # number of input nodes for fully connected part (TODO make dynamic using window size and img size)
numTestData = 100  # number of images to test on
numTrainData = 500 # number of images to train on
maskUsed = mask3   # mask that is used in the convolution
windowsize = 3     # window size of the pooling layer in the CNN
stridesize = 3     # stride size of the pooling layer in the CNN

# randomly initialize our weights with mean 0
# weights from input layer ("output of conv. layer" nodes) to hidden layer (4 nodes) fully connected
syn0 = 2 * np.random.random((inputlayer, 80)) - 1

# weights from hidden layer (4 nodes) to output layer ("number of classes" nodes)
syn1 = 2 * np.random.random((80, len(allclasses))) - 1


for j in range(numTrainData):

    x = labelled_data[j]
    im = x[1] # input data (image) 
    y = x[2]  # desired (target) output

    conv = convLayer(im, maskUsed)
    reLu = reLuLayer(conv)
    pool, places = maxPoolLayer(reLu, windowsize, stridesize)
    #print(places)
    l0 = pool.reshape(1,sum(len(x) for x in pool))
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    l2_error = y - l2

    if (j % 1) == 0:
      print ("Error " + str(j) + " :" + str(np.mean(np.abs(l2_error))) + ", certainty: "+str(np.max(l2)) + ", correct: "+ str(np.argmax(l2)==np.argmax(y)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, deriv=True)
    
    # get error of pool layer
    pool_error = l1_delta.dot(syn0.T)

    # map maximal places to input of pooling layer and to pool_error. returns vector representations
    pool_input_vec, pool_input_errorvec = invertPool(reLu, places, pool_error)

    # pool does not contribute on error: pass on to ReLu layer
    relu_delta = pool_input_errorvec * reLuLayer(pool_input_vec, deriv=True)
    
    # conv_error = relu_delta.dot(....
    # conv_delta = ...

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    # usedMask += ....

totCorrect = 0
tot = 0
for j in range(numTrainData,numTestData+numTrainData):
  x = labelled_data[j]
  tot = tot + 1
  im = x[1]
  y = x[2]
  conv =  convLayer(im, maskUsed)
  reLu = reLuLayer(conv)
  pool, places = maxPoolLayer(reLu, windowsize, stridesize)
  l0 = pool.reshape(1,sum(len(x) for x in pool))
  l1 = nonlin(np.dot(l0, syn0))
  l2 = nonlin(np.dot(l1, syn1))
  if(np.argmax(l2)==np.argmax(y)):
    totCorrect = totCorrect + 1
  print("Percentage correct: "+str(float(totCorrect)/float(tot))+" ("+str(totCorrect)+"/"+str(tot)+"), certainty: "+str(np.max(l2)))
