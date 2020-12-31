#import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt 
import numpy as np
import argparse

#construct the argument pars and pars the argument 
ap = argparse.ArgumentParser()
ap.add_argument("-o" , "--output", required=True,
    help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_mldata("MNIST Original")

# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits
data = dataset.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data,
    dataset.target, test_size=0.25)

#convert the labels from integers to vectors or one-hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#define the 784-256-128-10 architecture using keras
model = Sequential()
model.add(Dense(256 , input_shape=(784,) , activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

#train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
    metrics=["accuracy"])
    H = model.fit(trainX, trainY , validation_data=(testX, testY),
        epochs=100, batch_size==128)
        