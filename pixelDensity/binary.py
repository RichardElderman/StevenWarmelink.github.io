import cv2
import numpy as np
from matplotlib import pyplot as plt





originalImage = cv2.imread("example.pgm")
staticBinary190 = originalImage
staticBinary100 = originalImage
img = cv2.imread('example.pgm',0)
tempimg = cv2.medianBlur(img,5)

width, hight, channels =originalImage.shape

for i in range(1,width):
    for j in range(1, hight):

        px = staticBinary190[i, j][0]
        if px < 190:
            staticBinary190[i, j] = 0
        else:
            staticBinary190 [i, j]=255

for i in range(1, width):
    for j in range(1, hight):

        px = staticBinary100[i, j][0]
        if px < 100:
            staticBinary100[i, j] = 0
        else:
            staticBinary100[i, j] = 255





ret,binaryThresh = cv2.threshold(originalImage,127,255,cv2.THRESH_BINARY)
ret,inversBinary = cv2.threshold(originalImage,127,255,cv2.THRESH_BINARY_INV)
ret,trunc = cv2.threshold(originalImage,127,255,cv2.THRESH_TRUNC)
ret,tozero = cv2.threshold(originalImage,127,255,cv2.THRESH_TOZERO)
ret,inverseTozero = cv2.threshold(originalImage,127,255,cv2.THRESH_TOZERO_INV)





meanAddaptive = cv2.adaptiveThreshold(tempimg,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,5)
gaussianAddaptive = cv2.adaptiveThreshold(tempimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)

# Otsu's thresholding
ret2,otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,otsu2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)





titles = ['Original Image','Binary','Binary Inverse','Trunc','Tozero','Inverse Tozero', 'Static at 190', 'Static at 100','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding','Outsu','Outsu After Gaussian']
images = [originalImage, binaryThresh, inversBinary, trunc, tozero, inverseTozero, staticBinary190, staticBinary100,meanAddaptive,gaussianAddaptive, otsu, otsu2 ]
#show all images sep
cv2.imshow("Static Threshold Set At 190 ",staticBinary190)
cv2.imshow("Static Threshold Set At 100 ",staticBinary100)
cv2.imshow('Binary', binaryThresh)
cv2.imshow('Binary Invers', inversBinary)
cv2.imshow('Trunc', trunc)
cv2.imshow('Tozero', tozero)
cv2.imshow('Tozero Inverse',inverseTozero)
cv2.imshow('Adaptive Mean Thresholding',meanAddaptive)
cv2.imshow('Adaptive Gaussian Thresholding',gaussianAddaptive)
cv2.imshow('Outsu', otsu)
cv2.imshow('Outsu After Gaussian', otsu2)

#show everything
for i in xrange(12):
    plt.subplot(6,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()






#save Image

cv2.imwrite('ResultImage.jpg', otsu2)
# close everything
cv2.waitKey(0)
cv2.destroyAllWindows()

