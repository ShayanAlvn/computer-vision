#import the necessory packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt 
import numpy as np
import argparse

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
    help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

#load the training and testing data, scale it into the range[0, 1],
#then reshape the design matrix
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY),(testX, testY))