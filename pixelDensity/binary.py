import cv2
import numpy as np
from matplotlib import pyplot as plt

originalImage = cv2.imread("example.pgm")

width, hight, channels =originalImage.shape

for i in range(1,width):
    for j in range(1, hight):
        
        px = originalImage[i, j][0]
        if px < 190:
            originalImage[i, j] = 0
        else:
            originalImage [i, j]=255


cv2.imshow("image",originalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()