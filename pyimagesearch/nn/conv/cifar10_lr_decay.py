#set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

#import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from minivggnet import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt 
import numpy as np
import argparse

def step_decay(epoch):
    #initialize the base initial learning rate, drop factor, and
    #epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    #compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    #return the learning rate
    return float(alpha)

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, 
    help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

#load the training and testing data, then scale it into the 
#range[0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]