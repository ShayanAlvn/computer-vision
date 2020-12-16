# penalty = 0

# for i in np.arange(0 , W.shape[0]):
#     for j in np.arange(0 , W.shape[1]):
#         penalty += (W[i][j] ** 2)

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.pre_processing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from imutils import paths
import argparse


#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

#grab the list of image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
