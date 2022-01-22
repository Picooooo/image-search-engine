import method.color as color
import method.hog as hog
import method.vgg16 as vgg16

def extract_img(img, method):
    extractor = None
    if method == "color":
        extractor = color.Color()
    elif method == "hog":
        extractor == hog.Hog()
    else:
        extractor = vgg16.Vgg16()

    feature = extractor.extract(img)

    return feature