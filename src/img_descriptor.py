import method.color
import method.hog
import method.vgg16

def extract_img(img, method):
    extractor = None
    if method == "color":
        extractor = Color()
    elif method == "hog":
        extractor == Hog()
    else:
        extractor = Vgg16()

    feature = extractor.extract(img)

    return feature