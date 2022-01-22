from skimage.feature import hog 
import cv2

class Hog:
    name = "Hog"

    def extract(self, img):
        fd, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    
        return fd
