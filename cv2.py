#chapter7
import cv2

class SimplePreprocessor:
    def __init__ (self , width , height , inter=cv2.INTER_AREA):
        
        self.width = width
        self.height = height
        self.inter = inter
    
    def preprocess (self , image):

        return cv2.resize(image , (self.width , self.height) , interpolation = self.inter)

#----------------------------------------------------------------------------------------------
import numpy as np
import os 

class SimpleDatasetLoader:
    def __init__ (self , preprocessoers=None):

        self.preprocessoers = preprocessoers

        if self.preprocessoers is None:
            self.preprocessoers = []

    def load(self , imagepaths , verbose =-1):
        data = []
        labels = []

        for (i , imagepath) in enumerate(imagepaths):
            image = cv2.imread(imagepath)
            label = imagepath.split(os.path.sep)[-2]

            if self.preprocessoers is not None:
                for p in self.preprocessoers:
                    image = p.preprocess(image)
            
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,len(imagePaths)))
                
        return (np.array(data), np.array(labels))