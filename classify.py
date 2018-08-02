from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-i", "--image", required=True)

COMPRESSION = 2	

args = vars(ap.parse_args())

image = cv2.imread(args["image"],0)
image = cv2.resize(image, (int(image.shape[1]/2),int(image.shape[0]/2)))
image = image.astype("float") / 255.0
image = img_to_array(image)
print(image.shape)
image = np.expand_dims(image, axis=0)
print(image.shape)
model = load_model(args["model"])
proba = model.predict(image)[0]
name="test"
proba=proba*255
proba = proba.astype(np.uint8)

ret,im_thresh0 = cv2.threshold(proba,190,255,cv2.THRESH_BINARY)
#im_thresh1 = cv2.adaptiveThreshold(proba,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
#im_thresh2= cv2.adaptiveThreshold(proba,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25,10)

cv2.imwrite(name+".png",proba)
cv2.imshow('binarized',proba)
cv2.waitKey(0)
cv2.destroyAllWindows()