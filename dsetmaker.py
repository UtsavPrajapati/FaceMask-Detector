# -*- coding: utf-8 -*-
"""
Created on Wed May 12 07:43:18 2021

@author: user
"""

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
from tensorflow.keras import backend
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle

import os
import cv2



datano = 0
imagefile="dataset"
modelfile ="dataset2.hdf5"


def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(imagefile))

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
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
    datano += 1
	# show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))


# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
print(set(labels))
print(datano)
labels = np_utils.to_categorical(labels, datano)
print("here")
# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
X_sparse = coo_matrix(data)
data, X_sparse, labels = shuffle(data, X_sparse, labels, random_state=0)
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labels, test_size=0.15, random_state=42)

# define the architecture of the network
model = Sequential()


model.add(Dense(768, input_dim=1024, kernel_initializer="uniform",activation="relu"))

model.add(Dense(384, activation="relu", kernel_initializer="uniform"))

model.add(Dense(datano))
model.add(Activation("softmax"))

# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.05)

print("this")
#model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(trainData, trainLabels, epochs=50, batch_size=128,
	verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save(modelfile)