#Importing requirment
import matplotlib.pyplot as plt
from random import shuffle
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from datetime import timedelta
from datetime import datetime
import math
import sys
import segmentation as seg

def loadUTF(path):

    readMe = open(path, 'r').readlines()

    utfs=[]
    for i in range(0, len(readMe)):
       temp=readMe[i].strip()
       utfs.append(temp)

    return utfs

def classifyImages(imgs, utfs):

    num_images = len(imgs)

    # allocate array for utf codes (strings)
    predictions = np.zeros(shape=num_images, dtype=np.int)

    # reshape input images to fit the CNN data structure
    images = np.vstack([x[1] for x in imgs])
        
    # labels = np.vstack([x[0] for x in test_data[i:j]])

    # Create a feed-dict with these images and labels.
    feed_dict = {x: images}
    # ,y_true: labels}
    predictions = session.run(y_pred_cls, feed_dict=feed_dict) ####################################

    print("Predicted classes: ")
    print(cls_pred)

    # convert class codes to utfs
    out = []
    for cl in classes:
        out.append(utfs[cl])

    return out


if __name__ =="__main__":

    # load list of utf codes (in the same order as the used model was trained on)
    utf_codes = loadUTF('Allclass_UTF.txt')

    meta_path = 'checkpoints/my_test_model-1000.meta'
    save_path = 'checkpoints/my-model.cktp'

    # load CNN
    print("\nTry to load model...")
    session=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(meta_path)  ##### need to change path
    saver.restore(session, save_path=save_path)

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    # w1 = graph.get_tensor_by_name("w1:0")
    # w2 = graph.get_tensor_by_name("w2:0")
    # feed_dict ={w1:13.0,w2:17.0}

    #Now, access the op that you want to run. (used in the classification function) 
    y_pred_cls = graph.get_tensor_by_name("output_to_restore:0") ### the name must be stated in the model that was saved
    print("Model loaded.")

    # process all raw input files
    for i, filename in enumerate(os.listdir(".")):
        if filename.endswith(".pgm"):
            i += 1
            print(filename, repr(i))
            # get list of cropped files, and list of lines in xml file
            images, xml_data = seg.loopthroughimages(filename, j)

            # classify cropped images, return list of utf codes
            pred_classes = classifyImages(images, utfcodes)

            # generate xml file for this image
            seg.createXMLFile(xml_data, pred_classes)

            # writeImages(square_images, readStr) ## maybe save every cropped image (filename=utf?) ?

