# Handwriting Recognition project

This repository contains the files required to run the automatic classification of Chinese handwritten characters for team CRITICAL. Run instructions and dependencies can be found below. The repository can be found [here](https://github.com/StevenWarmelink/StevenWarmelink.github.io "classifier repository webpage"), and the entire project (including training scripts etc.) can be found on our [main repository](https://github.com/HiradEmami/smartHWR "main repository webpage") page. 

### Dependencies
```
- Python version 3 or later (Python 3.5.2 was used for the project)
- Tensorflow
- Numpy
- OpenCV2 
```

### Run instructions

0. Clone the repository using the command `git clone https://github.com/StevenWarmelink/StevenWarmelink.github.io.git` in your preferred directory.
1. From the repository root folder, enter the `project` folder.
2. Put the folder with all images you wish to analyze in this folder. 
3. Run the simulation using the commando `python code/PIPELINE.py [folder_name]`, where [folder_name] is the name of the folder with images to be analyzed.
4. After the classification process is complete, the XML files with classifications can be found in the folder with the images.