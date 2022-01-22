from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2 as cv2

class Vgg16:
    name = "Vgg16"
    
    def extract(self, img):
        model = VGG16(weights='imagenet', include_top=True)

        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)

        features = model.predict(img_arr)

        return features
    
    def get_name_method(self):
        name = "Vgg16"
        return name
