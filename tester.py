# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:26:12 2021

@author: user
"""

import cv2
import numpy as np
from imutils import paths
from tensorflow.keras.models import load_model
import os

imagefile="test"
model = load_model("dataset2.hdf5")

data = []
labels = []

predictions=[]

mask_positive = 0;
mask_negative = 0;
maskless_pos = 0;
maskless_neg = 0;

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

image = cv2.imread("test.jpg",0)

classes = ["mask","maskless"]
imagePaths = list(paths.list_images(imagefile))

print(classes.index("mask"))
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath,0)

    label = imagePath.split(os.path.sep)[-1].split(".")[0].split("-")[0]

	# construct a feature vector raw pixel intensities, then update
	# the data matrix and labels lis
    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)
    
for x in range(0,len(data)):
    image = data[x]
    
    features = np.array([image])
    
    probs = model.predict(features)[0]
                
    prediction = probs.argmax(axis=0)
    #print(prediction)
    predictions.append(prediction)
    
    


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

nlabels=[classes.index(x) for x in labels]

cm = confusion_matrix(nlabels,predictions)
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes)

disp = disp.plot()


plt.show()

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(nlabels, predictions)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(nlabels, predictions)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(nlabels, predictions)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(nlabels, predictions)
print('F1 score: %f' % f1)