import numpy as np
import cv2 as cv2

class Color():
    name = "Color"

    def extract(self, img):
        mask = None
        bins = [8, 8, 8]

        lab = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0,1,2], mask, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        return hist


    
