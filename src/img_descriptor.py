from tensorflow.keras.preprocessing import image
import method.color as color
import method.hog as hog
import method.vgg16 as vgg16
import cv2

def load_image(img_path, target_size):
  img = cv2.imread(img_path)
  if target_size == -1:
    return img
  return cv2.resize(img, target_size)
  
def extract_img(img_path, method):

    if type(img_path) is str:
      if method == "color":
          img = load_image(img_path, -1)
      elif method == "hog":
          img = load_image(img_path, (224, 224))
      elif method == 'vgg16':
          img = load_image(img_path, (224, 224))
    else:
      img = img_path
      if method == 'hog':
        img = cv2.resize(img, (224, 224))
      elif method == 'vgg16':
        img = cv2.resize(img, (224, 224))

    extractor = None
    if method == "color":
        extractor = color.Color()
    elif method == "hog":
        extractor = hog.Hog()
    elif method == 'vgg16':
        extractor = vgg16.Vgg16()

    feature = extractor.extract(img)

    return feature