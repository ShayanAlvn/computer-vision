import cv2
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
                print("[INFO] processed {}/{}".format(i + 1,len(imagepaths)))
                
        return (np.array(data), np.array(labels))