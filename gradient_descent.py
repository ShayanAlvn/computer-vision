# while True:
#     Wgradient = evaluate_gradient(los ,  data , W)
#     W += -alpha * Wgradient

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activatoin(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))

def predict(X , W):
    # take the dot product between our features and weight matrix
    preds = sigmoid_activatoin(X.dot(W))

    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds

#construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-e" , "--epochs" , type=float , default=100 , help="# of epochs")
ap.add_argument("-a" , "--alpha" , type=float , default=0.01 , help="learning rate")
args = vars(ap.parse_args())