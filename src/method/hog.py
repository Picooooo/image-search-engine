from skimage.feature import hog 

class Hog:
    name = "Hog"

    def extract(self, img):
        resized_img = cv2.resize(img, (128, 128))
        fd, hog_img = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    
        return fd
