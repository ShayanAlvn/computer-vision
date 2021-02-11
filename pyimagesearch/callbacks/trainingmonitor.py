#import the necessary packages
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt 
import numpy as np 
import json
import os

class TrainigMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        #store the output path for the figure, the path to the JSON
        #serialized file, and the starting epoch
        super(TrainigMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        #initialize the history dictionary
        self.H = {}

        #if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                #check to see if a starting epoch was supplied
                if self.startAt > 0:
                    #loop over the entries in the history log and
                    #trim any entries that are past the starting 
                    #epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]