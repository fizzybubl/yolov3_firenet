import os
import glob
import cv2
path = 'data/images/*.jpg'

image_path = glob.glob(path)

for image in  image_path:
    img = cv2.imread(image,cv2.IMREAD_COLOR)
    cv2.startWindowThread()
    cv2.imshow('image', img)
    cv2.waitKey()
    print(image)
