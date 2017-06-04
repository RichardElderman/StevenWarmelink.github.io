from __future__ import print_function
from random import shuffle
import cv2
import os
import numpy as np

#Reads all of the labelled images
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

# generate target output for an image, array of zeros with 1 at correct class
def getDesiredOutput(length, onepos):
  out = np.zeros(shape=(1,length))
  out[0,onepos] = 1
  return out

# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

def reLuMatrix(x, deriv = False):
  # set every negative value in x to zero
    # use softplus method to be able to find derivative, which is nonlin with deriv=true (needed for learning)
    if(deriv == True):
        return nonlin(x, True)    
    return np.log(1+np.exp(x))

def supersample(dat, orgX, orgY, winSize, strideSize):
  # invert pooling, by reconstructing original matrix (currently copy max value at all places in window)
  out = np.zeros(shape=(orgX, orgY))
  datX, datY = np.shape(dat)
  for r in range(datX):
    for c in range(datY):
      out[r*strideSize : min((r+winSize)*strideSize, orgX), c*strideSize : min((c+winSize)*strideSize, orgY)] = dat[r,c]
  return out

def pooling(dat, window, stride):
  # slide window over dat in steps of stride, take in each window the maximum value
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

def convolution(dat, mask):
  return np.mean(np.multiply(dat, mask))

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

imgWidth = 128
imgHeight = 128


np.random.seed(1)


# randomly initialized masks, currently not updated yet (hence bad classifications?)
mask = np.random.random((2, 2)) - 1
mask3 = np.random.random((3, 3)) - 1
#mask3 = np.matrix([[1,-1,-1],
           #[-1,1,-1],
           #[-1,-1,1]])
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

# read imgs to classify, and get number of classes
labelled_data, allclasses = readLabelledData(1000)
print("Number of classes: "+str(len(allclasses)))
# shuffle images, such that classes are more evenly distributed across data set
shuffle(labelled_data)

# list of parameters
inputlayer = 1849
numTestData = 100
numTrainData = 500
maskUsed = mask3
windowsize = 3
stridesize = 3

# randomly initialize our weights with mean 0
# weights from input layer ("output of conv. layer" nodes) to hidden layer (4 nodes) fully connected
syn0 = 2 * np.random.random((inputlayer, 80)) - 1

# weights from hidden layer (4 nodes) to output layer ("number of classes" nodes)
syn1 = 2 * np.random.random((80, len(allclasses))) - 1


for j in range(numTrainData):

    x = labelled_data[j]
    im = x[1] # input data (image) 
    y = x[2]  # desired (target) output

    conv =  convLayer(im, maskUsed)
    reLu = reLuMatrix(conv)
    pool, places = pooling(reLu, windowsize, stridesize)
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

    only_maxes = np.multiply(reLu,places) # replace all irrelevant values (nonmax) with zero
    only_maxes_vec = only_maxes.reshape(1,sum(len(x) for x in only_maxes))
    # max_indices = only_maxes_vec[np.where( only_maxes_vec==1)]
    # merge_maxes = np.array((max_indices, pool_error)).T

    # reconstruct input matrix of pool layer, with errors at max places
    pool_error_fullmat = np.zeros(shape=(1,len(only_maxes_vec)))
    error_indx = 0
    for indx in range(len(only_maxes_vec)):
      if only_maxes_vec[0, indx]==1:
        pool_error_fullmat[0, indx] = pool_error[0, error_indx]
        error_indx +=1

    # pool does not contribute on error: pass on to ReLu layer
    
    relu_delta = pool_error_fullmat * nonlin(only_maxes_vec, deriv=True)
    
    #conv_error = relu_delta.dot(
    

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

totCorrect = 0
tot = 0
for j in range(numTrainData,numTestData+numTrainData):
  x = labelled_data[j]
  tot = tot + 1
  im = x[1]
  y = x[2]
  conv =  convLayer(im, maskUsed)
  reLu = reLuMatrix(conv)
  pool, places = pooling(reLu, windowsize, stridesize)
  l0 = pool.reshape(1,sum(len(x) for x in pool))
  l1 = nonlin(np.dot(l0, syn0))
  l2 = nonlin(np.dot(l1, syn1))
  if(np.argmax(l2)==np.argmax(y)):
    totCorrect = totCorrect + 1
  print("Percentage correct: "+str(float(totCorrect)/float(tot))+" ("+str(totCorrect)+"/"+str(tot)+"), certainty: "+str(np.max(l2)))
