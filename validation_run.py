#Importing requirment
import tensorflow as tf
import numpy as np
import cv2
import os
# import pipe_segmentation as seg

def binarize(img):
  # Performs gaussian blurring with a kernel size of (5,5)
  blur = cv2.GaussianBlur(img,(5,5),0)
  # Performs Otsu thresholding (binarization) on the blurred image
  ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # switch black/white index (such that white=0), and set black to 1 instead of 256
  otsu[otsu==0] = 1
  otsu[otsu==255] = 0
  return otsu

#Reads all of the labelled images, and determined all different classes
def readLabelledData(path):

  labelled_data = []
  for i, file in enumerate(sorted(os.listdir(path))):
    print(i,  end="\r")
    if file.endswith('.pgm'):
      img = cv2.imread(path+'/' + file, 0)
      img = img.reshape(1, sum(len(x) for x in img))
      bin_img = binarize(img) # for now binarize image here (must already be done, but not)
      #print(np.unique(bin_img))
      utf = file[:4]
      output, class_number = getDesiredOutput(utf)
      labelled_data.append((output,bin_img, utf, class_number))
  return labelled_data

# Generate target output for an image
# Args: length of target vector, index in vector that must have value 1
# Returns vector of zeros of given length, with 1 at given position
def getDesiredOutput(utf):
  out = np.zeros(shape=(1,len(utf_codes)))
  classnumber = utf_codes.index(utf)
  out[0,classnumber] = 1
  return out, classnumber

def loadUTF(path):
    readMe = open(path, 'r').readlines()
    utfs=[]
    for i in range(0, len(readMe)):
       temp=readMe[i].strip()
       utfs.append(temp)

    return utfs



def print_test_accuracy():

    test_batch_size = 100
    # Number of images in the test-set.
    num_test = len(validation_data)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = np.vstack([x[1] for x in validation_data[i:j]])
        # print(np.shape(images))
        # Get the associated labels.
        labels = np.vstack([x[0] for x in validation_data[i:j]])
        # print(labels[1])
        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}
        # print(feed_dict[x][0], feed_dict[y_true][0])
        # Calculate the predicted class using TensorFlow.
        # print(feed_dict[x][0], feed_dict[y_true][0])
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = np.hstack([x[3] for x in validation_data])
    print("Predicted classes: ")
    print(cls_pred)
    print("True classes: ")
    print(cls_true)
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    # print(correct)
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Validation-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))



    return acc

if __name__ =="__main__":

    meta_path = 'checkpoints/my-model.cktp.meta'
    model_path = 'checkpoints/my-model.cktp'
    images_path = "unbalanced data"
    utf_path = 'Allclass_UTF.txt'

    # load list of utf codes (in the same order as the used model was trained on)
    utf_codes = loadUTF(utf_path)

    print("Load validation data")
    validation_data = readLabelledData(images_path)
    print("Validation data loaded")

    # load CNN
    print("\nTry to load model...")
    session=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(meta_path) 
    saver.restore(session, save_path=model_path)

    graph = tf.get_default_graph()

    #Now, access the ops that you want to run. (used in the classification function) 
    y_pred_cls = graph.get_tensor_by_name("output_to_restore:0") ### the name must be stated in the model that was saved
    y_true = graph.get_tensor_by_name("y_true:0")
    x = graph.get_tensor_by_name("x:0") ### the name must be stated in the model that was saved

    print("Model loaded.")

    print_test_accuracy()
