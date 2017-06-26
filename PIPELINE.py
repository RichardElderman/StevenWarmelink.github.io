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
import pipe_segmentation as seg

def loadUTF(path):
    readMe = open(path, 'r').readlines()
    utfs=[]
    for i in range(0, len(readMe)):
       temp=readMe[i].strip()
       utfs.append(temp)

    return utfs

def classifyImages(imgs, utfs):

    num_images = len(imgs)

    # allocate array for class predictions
    predictions = np.zeros(shape=num_images, dtype=np.int)

    # reshape input images to fit the CNN data structure
    images = np.vstack([np.hstack(x) for x in imgs])

    # labels = np.vstack([x[0] for x in test_data[i:j]])

    # Create a feed-dict with these images and labels.
    feed_dict = {x : images} ########## x is a name defined in the model

    predictions = session.run(y_pred_cls, feed_dict=feed_dict) 

    # convert class codes to utfs
    out = []
    for cl in predictions:
        out.append(utfs[cl])

    return out


if __name__ =="__main__":

    # load list of utf codes (in the same order as the used model was trained on)
    utf_codes = loadUTF('Allclass_UTF.txt')

    meta_path = 'checkpoints/my-model.cktp.meta'
    model_path = 'checkpoints/my-model.cktp'
    images_path = "unlabelled"

    # load CNN
    print("\nTry to load model...")
    session=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(meta_path) 
    saver.restore(session, save_path=model_path)

    graph = tf.get_default_graph()

    #Now, access the ops that you want to run. (used in the classification function) 
    y_pred_cls = graph.get_tensor_by_name("output_to_restore:0") ### the name must be stated in the model that was saved
    x = graph.get_tensor_by_name("x:0") ### the name must be stated in the model that was saved

    print("Model loaded.")
    # process all raw input files
    for i, filename in enumerate(os.listdir(images_path)):
        if filename.endswith(".pgm"):
            i += 1
            print("Processing", filename)
            # get list of cropped files, and list of data for xml file
            images, xml_data = seg.loopthroughimages(images_path+"/"+filename)
            print("Segmentation completed, classifying...")
            # classify cropped images, return list of utf codes
            pred_classes = classifyImages(images, utf_codes)
            print("Classification completed")
            # generate xml file for this image
            seg.createXMLFile(xml_data, pred_classes)
            print("XML generated")
            print("Processing", filename, "finished")

            # writeImages(square_images, readStr) ## maybe save every cropped image (filename=utf?) ?

