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


    if(len(sys.argv)!=3): # check if args were given (argv includes pipeline.py, tehrefore 3)
        print("Wrong input pattern for this program.")
        print("The correct pattern is:")
        print("python pipeline.py some_input_image.pgm some_input_xml.xml")
        sys.exit(1)

    input_image = sys.argv[1]
    input_xml = sys.argv[2]
    assert(input_image.endswith(".pgm"))
    assert(input_xml.endswith(".xml"))

    image_name = input_image[:-4]
    xml_name = input_xml[:-4]
    assert(image_name==xml_name)

    if(os.path.isfile(input_image)==False):
        print("Error: input image file not found")
        sys.exit(1)

    # load list of utf codes (in the same order as the used model was trained on)
    utf_codes = loadUTF('Allclass_UTF.txt')

    meta_path = 'checkpoints/my-model.cktp.meta'
    model_path = 'checkpoints/my-model.cktp'

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

    # process input image file
    print("Processing", input_image)
    # get list of cropped files, and list of data for xml file
    images, xml_data = seg.loopthroughimages(input_image)
    print("Segmentation completed,", len(images), "characters were found")
    print("Classifying...")
    # classify cropped images, return list of utf codes
    if(len(images)>0):
        pred_classes = classifyImages(images, utf_codes)
        print("Classification completed")
        # generate xml file for this image
        seg.createXMLFile(xml_data, pred_classes, xml_name)
        print("XML generated")
        print("Processing", input_image, "finished")
    else:
        print("No characters were found in", input_image)
