#Importing requirment
import matplotlib.pyplot as plt
from random import shuffle
import tensorflow as tf
import numpy as np
import cv2
import os
# from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
# from tensorflow.examples.tutorials.mnist import input_data



#   Follow the link for tutorial :
#   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb

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
def readLabelledData(maxFiles):
  lbls = []
  for i, file in enumerate(sorted(os.listdir('labelled'))):
    if file.endswith('.pgm'):
      utf = file[:4]
      if utf not in lbls:
        lbls.append(utf)
      if i==maxFiles:
        break;

  numlbl = len(lbls) # number of classes in the data set
  labelled_data = []
  for i, file in enumerate(sorted(os.listdir('labelled'))):
    print(i,  end="\r")
    if file.endswith('.pgm'):
      img = cv2.imread('labelled/' + file, 0)
      img = img.reshape(1, sum(len(x) for x in img))
      bin_img = binarize(img) # for now binarize image here (must already be done, but not)
      #print(np.unique(bin_img))
      utf = file[:4]
      class_number = lbls.index(utf)
      output = getDesiredOutput(numlbl, lbls.index(utf))
      labelled_data.append((output,bin_img, utf, class_number))
    if i==maxFiles:
      break;
  return labelled_data, lbls

def createSubData(labelled_data, train_n, test_n, valid_n):
    train_set = []
    test_set = []
    valid_set = []
    for i, (output, image, utf, class_number) in enumerate(labelled_data):
        if(i < train_n):
            train_set.append((output, image, utf, class_number))
            continue
        if(i < train_n+test_n):
            test_set.append((output, image, utf, class_number))
            continue
        if(i < train_n+test_n+valid_n):
            valid_set.append((output, image, utf, class_number))
            continue
    return train_set, test_set, valid_set

# Generate target output for an image
# Args: length of target vector, index in vector that must have value 1
# Returns vector of zeros of given length, with 1 at given position
def getDesiredOutput(length, onepos):
  out = np.zeros(shape=(1,length))
  out[0,onepos] = 1
  return out

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def plot_images(images, cls_true, cls_pred=None):
    # print(len(images), len(cls_true))
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        start = total_iterations*train_batch_size
        end = min((total_iterations+1)*train_batch_size, len(train_data))

        x_batch = np.vstack([x[1] for x in train_data[start:end]])
        y_true_batch = np.vstack([x[0] for x in train_data[start:end]])
        # x_batch, y_true_batch = train_data.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 1 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = np.vstack([x[1] for x in test_data])
    images = images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    classes = np.vstack([x[3] for x in test_data])
    cls_true = classes[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(test_data)

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
        images = np.vstack([x[1] for x in test_data[i:j]])
        # print(np.shape(images))
        # Get the associated labels.
        labels = np.vstack([x[0] for x in test_data[i:j]])
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
    cls_true = np.hstack([x[3] for x in test_data])
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
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)





if __name__ =="__main__":
    # Configuration of CNN
    # Convolutional Layer 1.
    filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
    num_filters1 = 16  # There are 16 of these filters.

    # Convolutional Layer 2.
    filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
    num_filters2 = 36  # There are 36 of these filters.

    # Fully-connected layer.
    fc_size = 128  # Number of neurons in fully-connected layer.

    train_batch_size = 100

    #Importing Input & Print data
    tot_n = 100000
    train_n = 8000
    test_n = 800
    valid_n = 7200
    labelled_data, allclasses = readLabelledData(tot_n)
    shuffle(labelled_data)
    train_data, test_data, valid_data = createSubData(labelled_data, train_n, test_n, valid_n)

    #Data Dimension
    # We know that Chinese images are 128 pixels in each dimension.
    img_size = 128

    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = 1

    # Number of classes, one class for each of 10 digits.
    num_classes = len(allclasses)

    #checking if Images are correct
    # Get the first images from the test-set.
    images = [x[1] for x in train_data[0:9]]

    # Get the true classes for those images.
    cls_true = [x[0] for x in train_data[0:9]]

    # Plot the images and labels using our helper-function above.
    # plot_images(images=images, cls_true=cls_true)

    #placeholder Varriable
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    y_true_cls = tf.argmax(y_true, dimension=1)



    #Conv Layer init
    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                       num_input_channels=num_channels,
                       filter_size=filter_size1,
                       num_filters=num_filters1,
                       use_pooling=True)

    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                       num_input_channels=num_filters1,
                       filter_size=filter_size2,
                       num_filters=num_filters2,
                       use_pooling=True)

    layer_flat, num_features = flatten_layer(layer_conv2)

    print("Number of Features: ")
    print(num_features)



    #Fully connect
    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=num_classes,
                             use_relu=False)

    #preditct
    y_pred = tf.nn.softmax(layer_fc2)
    print(y_pred)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    #cross entropy . Softmax
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                            labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    #oPTIMIZATION
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    #Correct Pre
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    ## next line == error ###############################################################################
    print_test_accuracy()

    optimize(num_iterations=20)

    print_test_accuracy()

    optimize(num_iterations=3)

    print_test_accuracy(show_example_errors=True)

    optimize(num_iterations=20)

    print_test_accuracy(show_example_errors=True)
