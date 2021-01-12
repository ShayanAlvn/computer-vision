#import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImageToArrayPreprocessor 
from simplepreprocessor import SimplePreprocessor 
from simpledatasetloader import SimpleDatasetLoader
from shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np 
import argparse

#construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-d" , "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
args = vars(ap.parse_args())