import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from module.autoencoder import autoencoderClass
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ap=argparse.ArgumentParser()
ap.add_argument("-d","--data",required=True)
ap.add_argument("-l","--labels",required=True)
ap.add_argument("-m","--premodel",required=True)
ap.add_argument("-p","--plot",type=str,default="plot.png")
args=vars(ap.parse_args())

EPOCHS = 50
INIT_LR = 1e-3
BS = 5
COMPRESSION = 2

data=[]
labels=[]
print("loading data")
dataPaths=sorted(list(paths.list_images(args["data"])))
labelPaths=sorted(list(paths.list_images(args["labels"])))

for dataPath in dataPaths:
	image=cv2.imread(dataPath,0)
	image = cv2.resize(image, (image.shape[0]/COMPRESSION, image.shape[1]/COMPRESSION))
	image=img_to_array(image)
	data.append(image)

for labelPath in labelPaths:
	image=cv2.imread(labelPath,0)
	image = cv2.resize(image, (image.shape[0]/COMPRESSION, image.shape[1]/COMPRESSION))
	image=img_to_array(image)
	labels.append(image)

data=np.array(data,dtype="float")
labels=np.array(labels,dtype="float")

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.2)

def f_measure(y_true,y_pred):
	return f1_score(y_true,y_pred)

print("loading model ")
model=load_model(args["premodel"])

for layer in model.layers[:3]:
	layer.trainable=False

opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,
	metrics=["accuracy"])

stop_here = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')


print("training ")
H = model.fit(trainX,trainY,
	validation_data=(testX,testY),
	batch_size=BS,
	epochs=EPOCHS,verbose=1)

model.save(model3.model)

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

