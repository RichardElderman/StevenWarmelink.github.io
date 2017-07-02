#Importing requirment
import tensorflow as tf
import numpy as np
import os
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

    # Create a feed-dict with these images and labels.
    feed_dict = {x : images} ########## x is a name defined in the model

    predictions = session.run(y_pred_cls, feed_dict=feed_dict) 

    # convert class codes to utfs
    out = []
    for cl in predictions:
        out.append(utfs[cl])

    return out


if __name__ =="__main__":


    if(len(sys.argv)!=2): # check if args were given (argv includes pipeline.py, tehrefore 3)
        print("Wrong input pattern for this program.")
        print("The correct pattern is:")
        print("python pipeline.py path_to_directory_with_raw_files")
        sys.exit(1)

    # check if argument is an existing folder
    input_dir = sys.argv[1]
    if(os.path.isdir(input_dir)==False):
        print("Image directory not found.")
        sys.exit(1)

    # load list of utf codes (in the same order as the used model was trained on)
    utf_codes = loadUTF('misc/Allclass_UTF.txt')

    # paths to the saved CNN files
    meta_path = 'misc/my-model.cktp.meta'
    model_path = 'misc/my-model.cktp'

    # load CNN
    print("\nTry to load model...")
    session=tf.Session()
    saver = tf.train.import_meta_graph(meta_path) # load meta graph to be able to call certain parts in the CNN
    saver.restore(session, save_path=model_path) # restore CNN

    graph = tf.get_default_graph()

    # assign a variable to the parts in the CNN that will be called in a function (input and output layer)
    y_pred_cls = graph.get_tensor_by_name("output_to_resotre:0") ### the name must be stated in the model that was saved
    x = graph.get_tensor_by_name("x:0") ### the name must be stated in the model that was saved

    print("Model loaded.")

    for i, file in enumerate(sorted(os.listdir(input_dir))):
        if(file.endswith(".pgm")):
            # process input image file
            print("\nProcessing", file)
            # get list of cropped files, and list of data for xml file
            images, xml_data = seg.loopthroughimages(input_dir+"/"+file)
            # classify cropped images, return list of utf codes
            if(len(images)>0):
                print("Segmentation completed,", len(images), "characters were found")
                print("Classifying...")
                pred_classes = classifyImages(images, utf_codes)
                print("Classification completed")
                # generate xml file for this image
                xml_name = input_dir+"/"+file[:-4]
                seg.createXMLFile(xml_data[0:len(pred_classes)], pred_classes, xml_name)
                print("XML generated")
                print("Processing", file, "finished")
            else:
                print("Segmentation completed, no characters were found in", file)
                print("Processing", file, "finished")
    print("\nFinished processing all pgm files in the folder \""+input_dir+"\"")
